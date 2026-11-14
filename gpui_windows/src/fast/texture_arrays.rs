//! BC7 and RGBA8 `Texture2DArray` storage for thumbnail sprites.
//!
//! The core crate owns the logical array. This store owns the Direct3D 11
//! texture and shader resource view the array sprite pipeline samples.

use std::{collections::HashMap, sync::Arc};

use anyhow::{Context as _, Result};
use gpui::{DevicePixels, Size, TextureArrayFormat, TextureArrayId, TextureArrayPlatform};
use parking_lot::Mutex;
use windows::Win32::Graphics::{
    Direct3D::D3D11_SRV_DIMENSION_TEXTURE2DARRAY,
    Direct3D11::{
        D3D11_BIND_SHADER_RESOURCE, D3D11_FORMAT_SUPPORT_TEXTURE2D,
        D3D11_SHADER_RESOURCE_VIEW_DESC, D3D11_TEX2D_ARRAY_SRV, D3D11_TEXTURE2D_DESC,
        D3D11_USAGE_DEFAULT, ID3D11Device, ID3D11DeviceContext, ID3D11ShaderResourceView,
        ID3D11Texture2D,
    },
    Dxgi::Common::{
        DXGI_FORMAT, DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
    },
};

/// The Direct3D format that stores layers of `format`, if it is supported.
fn dxgi_texture_format(format: TextureArrayFormat) -> Option<DXGI_FORMAT> {
    match format {
        TextureArrayFormat::Bc7RgbaUnorm => Some(DXGI_FORMAT_BC7_UNORM),
        TextureArrayFormat::Rgba8Unorm => Some(DXGI_FORMAT_R8G8B8A8_UNORM),
        _ => None,
    }
}

/// The bytes per row and per layer of one `format` layer with these
/// dimensions.
fn layer_row_layout(format: TextureArrayFormat, width: u32, height: u32) -> Result<(u32, u32)> {
    match format {
        TextureArrayFormat::Bc7RgbaUnorm => {
            anyhow::ensure!(
                width.is_multiple_of(4) && height.is_multiple_of(4),
                "BC7 array dimensions must be multiples of four"
            );
            let row_pitch = (width / 4)
                .checked_mul(16)
                .context("texture array row pitch overflow")?;
            let depth_pitch = row_pitch
                .checked_mul(height / 4)
                .context("texture array depth pitch overflow")?;
            Ok((row_pitch, depth_pitch))
        }
        TextureArrayFormat::Rgba8Unorm => {
            let row_pitch = width
                .checked_mul(4)
                .context("texture array row pitch overflow")?;
            let depth_pitch = row_pitch
                .checked_mul(height)
                .context("texture array depth pitch overflow")?;
            Ok((row_pitch, depth_pitch))
        }
        _ => anyhow::bail!("unsupported texture array format"),
    }
}

struct GpuArray {
    texture: ID3D11Texture2D,
    view: ID3D11ShaderResourceView,
    /// How the uploaded layers are laid out, which decides the upload pitches.
    format: TextureArrayFormat,
}

/// The arrays one Direct3D device owns.
pub(crate) struct TextureArrayStore {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    arrays: Mutex<HashMap<TextureArrayId, GpuArray>>,
}

impl TextureArrayStore {
    pub(crate) fn new(device: ID3D11Device, context: ID3D11DeviceContext) -> Arc<Self> {
        Arc::new(Self {
            device,
            context,
            arrays: Mutex::new(HashMap::new()),
        })
    }

    /// The shader resource view used to sample `id`, if the array is alive.
    pub(crate) fn view(&self, id: TextureArrayId) -> Option<ID3D11ShaderResourceView> {
        self.arrays.lock().get(&id).map(|array| array.view.clone())
    }

    fn create_gpu_texture(
        &self,
        id: TextureArrayId,
        size: Size<DevicePixels>,
        layers: u32,
        format: TextureArrayFormat,
    ) -> Result<()> {
        let dxgi_format =
            dxgi_texture_format(format).context("unsupported texture array format")?;
        let support = unsafe { self.device.CheckFormatSupport(dxgi_format) }
            .context("cannot query texture array format support")?;
        anyhow::ensure!(
            support & D3D11_FORMAT_SUPPORT_TEXTURE2D.0 as u32 != 0,
            "texture arrays of {format:?} are not supported by this device"
        );
        let width = u32::try_from(size.width.0).context("negative array width")?;
        let height = u32::try_from(size.height.0).context("negative array height")?;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: layers,
            Format: dxgi_format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture: Option<ID3D11Texture2D> = None;
        unsafe {
            self.device
                .CreateTexture2D(&desc, None, Some(&mut texture))
                .context("cannot create texture array")?;
        }
        let texture = texture.context("device returned no texture array")?;
        let view_desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
            Format: dxgi_format,
            ViewDimension: D3D11_SRV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: windows::Win32::Graphics::Direct3D11::D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D11_TEX2D_ARRAY_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    FirstArraySlice: 0,
                    ArraySize: layers,
                },
            },
        };
        let mut view: Option<ID3D11ShaderResourceView> = None;
        unsafe {
            self.device
                .CreateShaderResourceView(&texture, Some(&view_desc), Some(&mut view))
                .context("cannot create texture array view")?;
        }
        let view = view.context("device returned no array view")?;
        self.arrays.lock().insert(
            id,
            GpuArray {
                texture,
                view,
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
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { array.texture.GetDesc(&mut desc) };
        anyhow::ensure!(layer < desc.ArraySize, "texture array layer out of range");
        anyhow::ensure!(
            dxgi_texture_format(array.format) == Some(desc.Format),
            "texture array format changed after creation"
        );
        let (row_pitch, depth_pitch) = layer_row_layout(array.format, desc.Width, desc.Height)?;
        anyhow::ensure!(
            blocks.len() >= depth_pitch as usize,
            "texture array layer is shorter than its dimensions"
        );
        unsafe {
            // `layer` already selects the array slice through DstSubresource.
            // The optional box is relative to that subresource, whose depth is
            // one, so passing a box would have to keep front=0/back=1. Passing
            // `None` updates the whole slice and avoids that trap.
            self.context.UpdateSubresource(
                &array.texture,
                layer,
                None,
                blocks.as_ptr().cast(),
                row_pitch,
                depth_pitch,
            );
        }
        Ok(())
    }

    fn remove_texture(&self, id: TextureArrayId) {
        self.arrays.lock().remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directx_devices::DirectXDevices;
    use gpui::{AtlasTextureId, AtlasTextureKind, size};
    use windows::Win32::Graphics::Direct3D11::{
        D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE, D3D11_USAGE_STAGING,
    };

    #[test]
    fn rgba8_uploads_arbitrary_dimensions() {
        let Ok(devices) = DirectXDevices::new() else {
            return;
        };
        let store = TextureArrayStore::new(devices.device.clone(), devices.device_context.clone());
        let id = TextureArrayId::from_atlas_texture_id(AtlasTextureId {
            index: 0xD000_0000 + 43,
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
        let pixels = [0x11u8; 3 * 2 * 4];
        store.upload_layer(id, 0, &pixels).expect("rgba8 upload");
        store.remove_texture(id);
    }

    #[test]
    fn formats_map_to_matching_dxgi_formats() {
        assert_eq!(
            dxgi_texture_format(TextureArrayFormat::Bc7RgbaUnorm),
            Some(DXGI_FORMAT_BC7_UNORM)
        );
        assert_eq!(
            dxgi_texture_format(TextureArrayFormat::Rgba8Unorm),
            Some(DXGI_FORMAT_R8G8B8A8_UNORM)
        );
    }

    #[test]
    fn upload_pitches_follow_the_array_format() {
        assert_eq!(
            layer_row_layout(TextureArrayFormat::Bc7RgbaUnorm, 8, 8).expect("bc7 layout"),
            (32, 64)
        );
        assert_eq!(
            layer_row_layout(TextureArrayFormat::Rgba8Unorm, 3, 2).expect("rgba8 layout"),
            (12, 24)
        );
        assert!(
            layer_row_layout(TextureArrayFormat::Bc7RgbaUnorm, 3, 2).is_err(),
            "BC7 layers need whole 4x4 blocks"
        );
    }

    #[test]
    fn uploads_each_layer_to_its_own_slice() {
        let Ok(devices) = DirectXDevices::new() else {
            return;
        };
        let store = TextureArrayStore::new(devices.device.clone(), devices.device_context.clone());
        let id = TextureArrayId::from_atlas_texture_id(AtlasTextureId {
            index: 0xD000_0000 + 1,
            kind: AtlasTextureKind::Polychrome,
        })
        .expect("array id");
        let array_size = size(DevicePixels(4), DevicePixels(4));
        if store
            .create_texture(id, array_size, 2, TextureArrayFormat::Bc7RgbaUnorm)
            .is_err()
        {
            return;
        }
        let zero = [0u8; 16];
        let ones = [0xffu8; 16];
        store.upload_layer(id, 0, &zero).expect("layer 0 upload");
        store.upload_layer(id, 1, &ones).expect("layer 1 upload");

        let desc = D3D11_TEXTURE2D_DESC {
            Width: 4,
            Height: 4,
            MipLevels: 1,
            ArraySize: 2,
            Format: DXGI_FORMAT_BC7_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
        };
        let mut staging: Option<ID3D11Texture2D> = None;
        unsafe {
            devices
                .device
                .CreateTexture2D(&desc, None, Some(&mut staging))
                .expect("create staging texture");
        }
        let staging = staging.expect("staging texture");
        {
            let arrays = store.arrays.lock();
            let array = arrays.get(&id).expect("array");
            unsafe {
                devices
                    .device_context
                    .CopyResource(&staging, &array.texture);
            }
        }

        for (layer, expected) in [(0, zero), (1, ones)] {
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            unsafe {
                devices
                    .device_context
                    .Map(&staging, layer, D3D11_MAP_READ, 0, Some(&mut mapped))
                    .expect("map staging layer");
                let bytes = std::slice::from_raw_parts(mapped.pData.cast::<u8>(), 16);
                assert_eq!(bytes, expected.as_slice(), "layer {layer} kept its upload");
                devices.device_context.Unmap(&staging, layer);
            }
        }
    }
}
