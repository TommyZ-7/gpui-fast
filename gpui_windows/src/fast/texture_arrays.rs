//! BC7 `Texture2DArray` storage for thumbnail sprites.
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
        D3D11_BIND_SHADER_RESOURCE, D3D11_BOX, D3D11_FORMAT_SUPPORT_TEXTURE2D,
        D3D11_SHADER_RESOURCE_VIEW_DESC, D3D11_TEX2D_ARRAY_SRV, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, ID3D11Device,
        ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D,
    },
    Dxgi::Common::{DXGI_FORMAT_BC7_UNORM, DXGI_SAMPLE_DESC},
};

struct GpuArray {
    texture: ID3D11Texture2D,
    view: ID3D11ShaderResourceView,
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
    ) -> Result<()> {
        let support = unsafe { self.device.CheckFormatSupport(DXGI_FORMAT_BC7_UNORM) }
            .context("cannot query BC7 texture support")?;
        anyhow::ensure!(
            support & D3D11_FORMAT_SUPPORT_TEXTURE2D.0 as u32 != 0,
            "BC7 texture arrays are not supported by this device"
        );
        let width = u32::try_from(size.width.0).context("negative array width")?;
        let height = u32::try_from(size.height.0).context("negative array height")?;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: layers,
            Format: DXGI_FORMAT_BC7_UNORM,
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
                .context("cannot create BC7 texture array")?;
        }
        let texture = texture.context("device returned no texture array")?;
        let view_desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_BC7_UNORM,
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
                .context("cannot create BC7 texture array view")?;
        }
        let view = view.context("device returned no array view")?;
        self.arrays.lock().insert(id, GpuArray { texture, view });
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
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { array.texture.GetDesc(&mut desc) };
        anyhow::ensure!(layer < desc.ArraySize, "texture array layer out of range");
        anyhow::ensure!(
            desc.Width % 4 == 0 && desc.Height % 4 == 0,
            "BC7 array dimensions must be multiples of four"
        );
        let row_pitch = (desc.Width / 4)
            .checked_mul(16)
            .context("texture array row pitch overflow")?;
        let depth_pitch = row_pitch
            .checked_mul(desc.Height / 4)
            .context("texture array depth pitch overflow")?;
        anyhow::ensure!(
            blocks.len() >= depth_pitch as usize,
            "texture array layer is shorter than its dimensions"
        );
        let destination = D3D11_BOX {
            left: 0,
            top: 0,
            front: layer,
            right: desc.Width,
            bottom: desc.Height,
            back: layer + 1,
        };
        unsafe {
            self.context.UpdateSubresource(
                &array.texture,
                layer,
                Some(&destination),
                blocks.as_ptr() as *const _,
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
