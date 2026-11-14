//! BC7 and RGBA8 `Texture2DArray` storage for thumbnail sprites.
//!
//! The core crate owns the logical array (size, layer count, versions). This
//! store owns the GPU texture and the bind group the array sprite pipeline
//! samples. It is created once per renderer device and installed as the
//! process-wide [`TextureArrayPlatform`].

use std::{collections::HashMap, sync::Arc};

use anyhow::{Context as _, Result};
use gpui::{DevicePixels, Size, TextureArrayFormat, TextureArrayId, TextureArrayPlatform};
use parking_lot::Mutex;

/// A GPU texture and the bind group that exposes it to the array pipeline.
struct GpuArray {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    /// How the uploaded layers are laid out, which decides the upload strides.
    format: TextureArrayFormat,
}

/// The arrays one renderer device owns.
pub(crate) struct TextureArrayStore {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    arrays: Mutex<HashMap<TextureArrayId, GpuArray>>,
}

pub(crate) fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("texture_array_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

pub(crate) fn sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("texture_array_sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

/// The wgpu format that stores layers of `format`, if it is supported.
fn wgpu_texture_format(format: TextureArrayFormat) -> Option<wgpu::TextureFormat> {
    match format {
        TextureArrayFormat::Bc7RgbaUnorm => Some(wgpu::TextureFormat::Bc7RgbaUnorm),
        TextureArrayFormat::Rgba8Unorm => Some(wgpu::TextureFormat::Rgba8Unorm),
        _ => None,
    }
}

/// The bytes per row and number of rows of one `format` layer with these
/// dimensions.
fn layer_row_layout(format: TextureArrayFormat, width: u32, height: u32) -> Result<(u32, u32)> {
    match format {
        TextureArrayFormat::Bc7RgbaUnorm => {
            anyhow::ensure!(
                width.is_multiple_of(4) && height.is_multiple_of(4),
                "BC7 array dimensions must be multiples of four"
            );
            let row_bytes = (width / 4)
                .checked_mul(16)
                .context("texture array row size overflow")?;
            Ok((row_bytes, height / 4))
        }
        TextureArrayFormat::Rgba8Unorm => {
            let row_bytes = width
                .checked_mul(4)
                .context("texture array row size overflow")?;
            Ok((row_bytes, height))
        }
        _ => anyhow::bail!("unsupported texture array format"),
    }
}

impl TextureArrayStore {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        layout: wgpu::BindGroupLayout,
        sampler: wgpu::Sampler,
    ) -> Arc<Self> {
        Arc::new(Self {
            device,
            queue,
            layout,
            sampler,
            arrays: Mutex::new(HashMap::new()),
        })
    }

    /// The layout every array pipeline must use for its bind group.
    pub(crate) fn layout(&self) -> wgpu::BindGroupLayout {
        self.layout.clone()
    }

    /// The bind group used to sample `id`, if the array is still alive.
    pub(crate) fn bind_group(&self, id: TextureArrayId) -> Option<wgpu::BindGroup> {
        self.arrays
            .lock()
            .get(&id)
            .map(|array| array.bind_group.clone())
    }

    fn create_gpu_texture(
        &self,
        id: TextureArrayId,
        size: Size<DevicePixels>,
        layers: u32,
        format: TextureArrayFormat,
    ) -> Result<()> {
        let wgpu_format =
            wgpu_texture_format(format).context("unsupported texture array format")?;
        if format == TextureArrayFormat::Bc7RgbaUnorm {
            anyhow::ensure!(
                self.device
                    .features()
                    .contains(wgpu::Features::TEXTURE_COMPRESSION_BC),
                "BC7 texture compression is not supported by this device"
            );
        }
        let width = u32::try_from(size.width.0).context("negative array width")?;
        let height = u32::try_from(size.height.0).context("negative array height")?;
        let limits = self.device.limits();
        anyhow::ensure!(
            layers <= limits.max_texture_array_layers,
            "texture array needs {layers} layers, device supports {}",
            limits.max_texture_array_layers
        );
        anyhow::ensure!(
            width <= limits.max_texture_dimension_2d && height <= limits.max_texture_dimension_2d,
            "texture array {width}x{height} exceeds the device's {} pixel limit",
            limits.max_texture_dimension_2d
        );
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture_array"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture_array_bind_group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.arrays.lock().insert(
            id,
            GpuArray {
                texture,
                bind_group,
                format,
            },
        );
        Ok(())
    }
}

impl TextureArrayPlatform for TextureArrayStore {
    fn create_texture(
        &self,
        id: TextureArrayId,
        size: Size<DevicePixels>,
        layers: u32,
        format: TextureArrayFormat,
    ) -> Result<()> {
        self.create_gpu_texture(id, size, layers, format)
    }

    fn upload_layer(&self, id: TextureArrayId, layer: u32, blocks: &[u8]) -> Result<()> {
        let arrays = self.arrays.lock();
        let array = arrays.get(&id).context("unknown texture array")?;
        let size = array.texture.size();
        anyhow::ensure!(
            layer < size.depth_or_array_layers,
            "texture array layer out of range"
        );
        let (row_bytes, rows_per_image) = layer_row_layout(array.format, size.width, size.height)?;
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &array.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            blocks,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(rows_per_image),
            },
            wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    fn remove_texture(&self, id: TextureArrayId) {
        self.arrays.lock().remove(&id);
    }
}

/// Creates the store owned by one `WgpuContext` and installs it as the
/// process-wide upload backend.
///
/// A context and its store are replaced together when the device is lost, so
/// this uses the replace path rather than first-wins installation.
pub(crate) fn init(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    install_platform: bool,
) -> Arc<TextureArrayStore> {
    let store = TextureArrayStore::new(
        Arc::clone(device),
        Arc::clone(queue),
        bind_group_layout(device),
        sampler(device),
    );
    if install_platform {
        gpui::replace_texture_array_platform(store.clone());
    }
    store
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{AtlasTextureId, AtlasTextureKind, DevicePixels, size};

    #[test]
    fn rgba8_array_accepts_arbitrary_dimensions() {
        let Ok((context, _)) = crate::WgpuContext::new_headless() else {
            return;
        };
        let store = init(&context.device, &context.queue, false);
        let id = TextureArrayId::from_atlas_texture_id(AtlasTextureId {
            index: 0xD000_0000 + 42,
            kind: AtlasTextureKind::Polychrome,
        })
        .expect("array id");
        let array_size = size(DevicePixels(3), DevicePixels(2));
        if store
            .create_texture(id, array_size, 1, TextureArrayFormat::Rgba8Unorm)
            .is_err()
        {
            return;
        }
        store
            .upload_layer(id, 0, &[0u8; 3 * 2 * 4])
            .expect("rgba8 upload");
        store.remove_texture(id);
    }

    #[test]
    fn rgba8_array_creation_succeeds_when_a_gpu_is_available() {
        let Ok((context, _)) = crate::WgpuContext::new_headless() else {
            return;
        };
        let store = init(&context.device, &context.queue, false);
        let id = TextureArrayId::from_atlas_texture_id(AtlasTextureId {
            index: 0xD000_0000 + 43,
            kind: AtlasTextureKind::Polychrome,
        })
        .expect("array id");
        store
            .create_texture(
                id,
                size(DevicePixels(3), DevicePixels(2)),
                1,
                TextureArrayFormat::Rgba8Unorm,
            )
            .expect("rgba8 array");
        store
            .upload_layer(id, 0, &[0u8; 3 * 2 * 4])
            .expect("rgba8 upload");
        store.remove_texture(id);
    }

    #[test]
    fn formats_map_to_matching_wgpu_texture_formats() {
        assert_eq!(
            wgpu_texture_format(TextureArrayFormat::Bc7RgbaUnorm),
            Some(wgpu::TextureFormat::Bc7RgbaUnorm)
        );
        assert_eq!(
            wgpu_texture_format(TextureArrayFormat::Rgba8Unorm),
            Some(wgpu::TextureFormat::Rgba8Unorm)
        );
    }

    #[test]
    fn upload_rows_follow_the_array_format() {
        assert_eq!(
            layer_row_layout(TextureArrayFormat::Bc7RgbaUnorm, 8, 8).expect("bc7 layout"),
            (32, 2)
        );
        assert_eq!(
            layer_row_layout(TextureArrayFormat::Rgba8Unorm, 3, 2).expect("rgba8 layout"),
            (12, 2)
        );
        assert!(
            layer_row_layout(TextureArrayFormat::Bc7RgbaUnorm, 3, 2).is_err(),
            "BC7 layers need whole 4x4 blocks"
        );
    }
}
