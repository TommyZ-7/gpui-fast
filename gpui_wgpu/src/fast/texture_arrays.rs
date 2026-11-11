//! BC7 `Texture2DArray` storage for thumbnail sprites.
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
}

/// The arrays one renderer device owns.
pub(crate) struct TextureArrayStore {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    arrays: Mutex<HashMap<TextureArrayId, GpuArray>>,
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
    ) -> Result<()> {
        anyhow::ensure!(
            self.device
                .features()
                .contains(wgpu::Features::TEXTURE_COMPRESSION_BC),
            "BC7 texture compression is not supported by this device"
        );
        let width = u32::try_from(size.width.0).context("negative array width")?;
        let height = u32::try_from(size.height.0).context("negative array height")?;
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
            format: wgpu::TextureFormat::Bc7RgbaUnorm,
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
        anyhow::ensure!(
            format == TextureArrayFormat::Bc7RgbaUnorm,
            "unsupported texture array format"
        );
        self.create_gpu_texture(id, size, layers)
    }

    fn upload_layer(&self, id: TextureArrayId, layer: u32, blocks: &[u8]) -> Result<()> {
        let arrays = self.arrays.lock();
        let array = arrays.get(&id).context("unknown texture array")?;
        let size = array.texture.size();
        anyhow::ensure!(
            layer < size.depth_or_array_layers,
            "texture array layer out of range"
        );
        anyhow::ensure!(
            size.width % 4 == 0 && size.height % 4 == 0,
            "BC7 array dimensions must be multiples of four"
        );
        let blocks_x = size.width / 4;
        let blocks_y = size.height / 4;
        let row_bytes = blocks_x
            .checked_mul(16)
            .context("texture array row size overflow")?;
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
                rows_per_image: Some(blocks_y),
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

/// Installs the array store as the process-wide upload backend.
pub(crate) fn init(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
) -> Arc<TextureArrayStore> {
    let store = TextureArrayStore::new(device, queue, layout, sampler);
    gpui::set_texture_array_platform(store.clone());
    store
}

