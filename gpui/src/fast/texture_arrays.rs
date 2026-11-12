//! Immutable, block-compressed texture arrays sampled as ordinary sprites.
//!
//! A [`TextureArray`] is a platform-owned `Texture2DArray`. Applications
//! upload whole layers of BC7 blocks and paint them with
//! [`Window::paint_texture_array_layer`]. The scene carries them as
//! [`PolychromeSprite`]s in a reserved atlas id range, so batching, clipping,
//! rounded corners and scroll-tile hashing reuse the existing sprite path.
//!
//! The platform backend is installed once by the renderer that owns the
//! device. Platforms without one (macOS/Metal in this fork) report an error
//! from [`Window::create_texture_array`]; there is no CPU fallback.

use anyhow::{Context as _, Result};
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
};

use crate::{
    AtlasTextureId, AtlasTextureKind, AtlasTile, Bounds, Corners, DevicePixels, Pixels,
    PolychromeSprite, Size, TileId, Window,
};

/// Atlas texture indices at or above this base belong to texture arrays.
///
/// Dynamic textures reserve `0xE000_0000` and layer tiles `0xF000_0000`, so
/// this range keeps the three id spaces disjoint without changing upstream's
/// `AtlasTextureKind`.
pub const TEXTURE_ARRAY_INDEX_BASE: u32 = 0xD000_0000;

/// How many texture array ids the reserved atlas index range holds.
const TEXTURE_ARRAY_ID_LIMIT: u32 = 0x1000_0000;

/// The pixel format of a texture array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureArrayFormat {
    /// BC7 compressed RGBA, one byte per pixel (16 bytes per 4x4 block).
    Bc7RgbaUnorm,
}

impl TextureArrayFormat {
    /// The exact byte length of one layer of `size`.
    pub fn layer_len(self, size: Size<DevicePixels>) -> Option<usize> {
        match self {
            Self::Bc7RgbaUnorm => {
                if size.width.0 <= 0
                    || size.height.0 <= 0
                    || size.width.0 % 4 != 0
                    || size.height.0 % 4 != 0
                {
                    return None;
                }
                let blocks_x = size.width.0 as usize / 4;
                let blocks_y = size.height.0 as usize / 4;
                blocks_x.checked_mul(blocks_y)?.checked_mul(16)
            }
        }
    }
}

/// How the stored texture is rotated when sampled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TextureArrayUv {
    /// The stored layer is upright in display space.
    Normal = 0,
    /// The stored layer was rotated 90° counter-clockwise before upload.
    Rotate90 = 1,
}

/// Identifies one texture array for its lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureArrayId(u32);

impl TextureArrayId {
    /// The atlas texture id a sprite of this array carries. Renderers resolve
    /// this reserved range before consulting the sprite atlas.
    pub fn atlas_texture_id(self) -> AtlasTextureId {
        AtlasTextureId {
            index: TEXTURE_ARRAY_INDEX_BASE.wrapping_add(self.0),
            kind: AtlasTextureKind::Polychrome,
        }
    }

    /// The texture array id behind an atlas texture id, if it is one.
    pub fn from_atlas_texture_id(id: AtlasTextureId) -> Option<Self> {
        (id.kind == AtlasTextureKind::Polychrome
            && id.index >= TEXTURE_ARRAY_INDEX_BASE
            && id.index - TEXTURE_ARRAY_INDEX_BASE < TEXTURE_ARRAY_ID_LIMIT)
            .then(|| Self(id.index - TEXTURE_ARRAY_INDEX_BASE))
    }
}

/// A copyable handle to an immutable texture array.
///
/// The array is owned by the renderer's backend; removing it retires every
/// copy of the handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureArray {
    id: TextureArrayId,
    size: Size<DevicePixels>,
    layers: u32,
    format: TextureArrayFormat,
}

impl TextureArray {
    /// Identifies the array in its renderer's lookup table.
    pub fn id(&self) -> TextureArrayId {
        self.id
    }

    /// The 2D size of every layer.
    pub fn size(&self) -> Size<DevicePixels> {
        self.size
    }

    /// The number of layers.
    pub fn layers(&self) -> u32 {
        self.layers
    }

    /// The layer format.
    pub fn format(&self) -> TextureArrayFormat {
        self.format
    }

    /// Creates a handle for unit tests that only exercise cache bookkeeping.
    ///
    /// The handle owns no GPU resource; painting or uploading it will fail
    /// because its id is not registered with a platform backend.
    #[doc(hidden)]
    pub fn new_for_test(size: Size<DevicePixels>, layers: u32) -> Self {
        Self {
            id: TextureArrayId(0),
            size,
            layers,
            format: TextureArrayFormat::Bc7RgbaUnorm,
        }
    }

    /// Releases the array and its GPU memory without requiring a live window.
    pub fn remove(self) {
        remove(self);
    }
}

/// The GPU side of texture arrays, implemented by platform renderers.
pub trait TextureArrayPlatform: Send + Sync {
    /// Allocates the array. No layer contents are uploaded yet.
    fn create_texture(
        &self,
        id: TextureArrayId,
        size: Size<DevicePixels>,
        layers: u32,
        format: TextureArrayFormat,
    ) -> Result<()>;

    /// Uploads one tightly packed layer of `format.layer_len(size)` bytes.
    fn upload_layer(
        &self,
        id: TextureArrayId,
        layer: u32,
        blocks: &[u8],
    ) -> Result<()>;

    /// Releases the array allocated for `id`.
    fn remove_texture(&self, id: TextureArrayId);
}

static PLATFORM: parking_lot::RwLock<Option<Arc<dyn TextureArrayPlatform>>> =
    parking_lot::RwLock::new(None);

/// Installs the platform backend for texture arrays.
///
/// Called once by the renderer while it is coming up. The first backend wins:
/// arrays must stay on the device that draws them.
pub fn set_texture_array_platform(backend: Arc<dyn TextureArrayPlatform>) {
    let mut platform = PLATFORM.write();
    if platform.is_none() {
        *platform = Some(backend);
    }
}

fn platform() -> Result<Arc<dyn TextureArrayPlatform>> {
    require_platform(PLATFORM.read().clone())
}

/// Split out so the missing-backend branch can be tested after a mock has
/// installed the process-global platform.
fn require_platform(
    platform: Option<Arc<dyn TextureArrayPlatform>>,
) -> Result<Arc<dyn TextureArrayPlatform>> {
    platform.context("texture arrays are not supported on this platform")
}

static NEXT_ARRAY_ID: AtomicU32 = AtomicU32::new(0);
static IDS_EXHAUSTED: AtomicBool = AtomicBool::new(false);

struct ArrayState {
    size: Size<DevicePixels>,
    layers: u32,
    format: TextureArrayFormat,
    /// Bumped when one layer is uploaded. Sprites carry it in
    /// [`PolychromeSprite::pad`], which the scroll-tile hashes cover.
    versions: Vec<AtomicU32>,
}

static LIVE_ARRAYS: parking_lot::Mutex<Option<HashMap<TextureArrayId, Arc<ArrayState>>>> =
    parking_lot::Mutex::new(None);

fn live_arrays() -> parking_lot::MappedMutexGuard<'static, HashMap<TextureArrayId, Arc<ArrayState>>>
{
    parking_lot::MutexGuard::map(LIVE_ARRAYS.lock(), |arrays| {
        arrays.get_or_insert_with(HashMap::new)
    })
}

fn live_state(id: TextureArrayId) -> Result<Arc<ArrayState>> {
    live_arrays()
        .get(&id)
        .cloned()
        .context("unknown texture array")
}

/// The version a sprite painted now must carry.
fn painted_version(id: TextureArrayId, layer: u32) -> Result<u32> {
    let state = live_state(id)?;
    state
        .versions
        .get(layer as usize)
        .map(|version| version.load(Ordering::Relaxed).max(1))
        .context("texture array layer out of range")
}

fn create(
    size: Size<DevicePixels>,
    layers: u32,
    format: TextureArrayFormat,
) -> Result<TextureArray> {
    anyhow::ensure!(
        size.width.0 > 0 && size.height.0 > 0,
        "texture array dimensions must be positive"
    );
    anyhow::ensure!(layers > 0, "texture array must have at least one layer");
    anyhow::ensure!(
        format.layer_len(size).is_some(),
        "texture array dimensions are not valid for {format:?}"
    );

    let backend = platform()?;
    let id = TextureArrayId(NEXT_ARRAY_ID.fetch_add(1, Ordering::Relaxed));
    if IDS_EXHAUSTED.load(Ordering::Relaxed) || id.0 >= TEXTURE_ARRAY_ID_LIMIT {
        IDS_EXHAUSTED.store(true, Ordering::Relaxed);
        anyhow::bail!("texture array ids are exhausted");
    }
    backend
        .create_texture(id, size, layers, format)
        .with_context(|| format!("failed to create texture array {id:?}"))?;

    live_arrays().insert(
        id,
        Arc::new(ArrayState {
            size,
            layers,
            format,
            versions: (0..layers).map(|_| AtomicU32::new(1)).collect(),
        }),
    );

    Ok(TextureArray {
        id,
        size,
        layers,
        format,
    })
}

fn upload(array: TextureArray, layer: u32, blocks: &[u8]) -> Result<()> {
    let state = live_state(array.id)?;
    anyhow::ensure!(
        layer < state.layers,
        "texture array layer out of range: {layer} >= {}",
        state.layers
    );
    let expected = state
        .format
        .layer_len(state.size)
        .context("invalid texture array dimensions")?;
    anyhow::ensure!(
        blocks.len() == expected,
        "texture array layer must be {expected} bytes, got {}",
        blocks.len()
    );

    platform()?
        .upload_layer(array.id, layer, blocks)
        .with_context(|| format!("failed to upload texture array layer {layer}"))?;

    if let Some(version) = state.versions.get(layer as usize) {
        version.fetch_add(1, Ordering::Relaxed);
    }
    Ok(())
}

/// Paints one layer into the scene.
///
/// `bounds` is the element's frame and `image_bounds` the (possibly larger)
/// fitted image. The visible intersection determines the sprite bounds and the
/// texture sub-rectangle, exactly like an atlas image's object fit.
pub(crate) fn paint(
    window: &mut Window,
    bounds: Bounds<Pixels>,
    image_bounds: Bounds<Pixels>,
    array: TextureArray,
    layer: u32,
    uv: TextureArrayUv,
    corner_radii: Corners<Pixels>,
) -> Result<()> {
    window.invalidator.debug_assert_paint();

    let size = live_state(array.id)?.size;
    let visible_bounds = bounds.intersect(&image_bounds);
    if visible_bounds.size.width <= Pixels::ZERO || visible_bounds.size.height <= Pixels::ZERO {
        return Ok(());
    }
    if image_bounds.size.width <= Pixels::ZERO || image_bounds.size.height <= Pixels::ZERO {
        return Ok(());
    }

    let tile = sub_tile(size, image_bounds, visible_bounds);
    let visible_bounds_snapped = window.snap_bounds(visible_bounds);
    let content_mask = window.snapped_content_mask();
    let corner_radii = corner_radii
        .clamp_radii_for_quad_size(visible_bounds.size)
        .scale(window.scale_factor());
    let version = painted_version(array.id, layer)?;

    window.next_frame.scene.insert_primitive(PolychromeSprite {
        order: 0,
        pad: version,
        grayscale: false.into(),
        bounds: visible_bounds_snapped,
        content_mask,
        corner_radii,
        tile: AtlasTile {
            texture_id: array.id.atlas_texture_id(),
            tile_id: TileId(layer),
            padding: uv as u32,
            bounds: tile,
        },
        opacity: window.element_opacity(),
    });

    Ok(())
}

/// Maps the visible part of a fitted image back to texture pixels.
fn sub_tile(
    size: Size<DevicePixels>,
    image_bounds: Bounds<Pixels>,
    visible_bounds: Bounds<Pixels>,
) -> Bounds<DevicePixels> {
    let x_offset_ratio =
        (visible_bounds.origin.x - image_bounds.origin.x) / image_bounds.size.width;
    let y_offset_ratio =
        (visible_bounds.origin.y - image_bounds.origin.y) / image_bounds.size.height;
    let width_ratio = visible_bounds.size.width / image_bounds.size.width;
    let height_ratio = visible_bounds.size.height / image_bounds.size.height;

    let tile_width = size.width.0;
    let tile_height = size.height.0;
    let origin_x = (x_offset_ratio * tile_width as f32).round() as i32;
    let origin_y = (y_offset_ratio * tile_height as f32).round() as i32;
    let width = (width_ratio * tile_width as f32).round() as i32;
    let height = (height_ratio * tile_height as f32).round() as i32;

    let clamped_origin_x = origin_x.clamp(0, tile_width);
    let clamped_origin_y = origin_y.clamp(0, tile_height);
    let clamped_width = width.min(tile_width - clamped_origin_x).max(0);
    let clamped_height = height.min(tile_height - clamped_origin_y).max(0);

    Bounds {
        origin: crate::point(DevicePixels(clamped_origin_x), DevicePixels(clamped_origin_y)),
        size: crate::size(DevicePixels(clamped_width), DevicePixels(clamped_height)),
    }
}

/// Releases an array and its GPU memory.
fn remove(array: TextureArray) {
    if live_arrays().remove(&array.id).is_some()
        && let Ok(backend) = platform()
    {
        backend.remove_texture(array.id);
    }
}

impl Window {
    /// Allocates an immutable texture array with `layers` layers of `size`.
    ///
    /// The array is created outside the sprite atlas and starts with
    /// unspecified contents. Upload each layer with
    /// [`Window::upload_texture_array_layer`] before painting it.
    pub fn create_texture_array(
        &mut self,
        size: Size<DevicePixels>,
        layers: u32,
        format: TextureArrayFormat,
    ) -> Result<TextureArray> {
        create(size, layers, format)
    }

    /// Uploads one tightly packed BC7 layer and marks the window for redraw.
    pub fn upload_texture_array_layer(
        &mut self,
        array: TextureArray,
        layer: u32,
        blocks: &[u8],
    ) -> Result<()> {
        upload(array, layer, blocks)?;
        self.refresh();
        Ok(())
    }

    /// Paints one layer at the current z-index and opacity.
    ///
    /// Call during the paint phase. `Rotate90` treats the stored layer as
    /// rotated 90° counter-clockwise and restores it on sampling.
    pub fn paint_texture_array_layer(
        &mut self,
        bounds: Bounds<Pixels>,
        image_bounds: Bounds<Pixels>,
        array: TextureArray,
        layer: u32,
        uv: TextureArrayUv,
        corner_radii: Corners<Pixels>,
    ) -> Result<()> {
        paint(self, bounds, image_bounds, array, layer, uv, corner_radii)
    }

    /// Releases a texture array and its GPU memory.
    pub fn remove_texture_array(&mut self, array: TextureArray) {
        remove(array);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArrayState, TEXTURE_ARRAY_INDEX_BASE, TextureArray, TextureArrayFormat, TextureArrayId,
        TextureArrayPlatform, TextureArrayUv, create, live_arrays, live_state,
        require_platform, upload,
    };
    use crate::{AtlasTextureKind, Bounds, DevicePixels, Pixels, Size, point, size};
    use anyhow::Result;
    use std::sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    };

    struct MockPlatform {
        created: parking_lot::Mutex<Vec<(TextureArrayId, Size<DevicePixels>, u32)>>,
        uploaded: parking_lot::Mutex<Vec<(TextureArrayId, u32, usize)>>,
        removed: parking_lot::Mutex<Vec<TextureArrayId>>,
        fail_upload: AtomicBool,
    }

    impl TextureArrayPlatform for MockPlatform {
        fn create_texture(
            &self,
            id: TextureArrayId,
            size: Size<DevicePixels>,
            layers: u32,
            _format: TextureArrayFormat,
        ) -> Result<()> {
            self.created.lock().push((id, size, layers));
            Ok(())
        }

        fn upload_layer(
            &self,
            id: TextureArrayId,
            layer: u32,
            blocks: &[u8],
        ) -> Result<()> {
            if self.fail_upload.load(Ordering::SeqCst) {
                anyhow::bail!("backend refused the upload");
            }
            self.uploaded.lock().push((id, layer, blocks.len()));
            Ok(())
        }

        fn remove_texture(&self, id: TextureArrayId) {
            self.removed.lock().push(id);
        }
    }

    static MOCK_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    fn shared_mock() -> &'static Arc<MockPlatform> {
        static MOCK: OnceLock<Arc<MockPlatform>> = OnceLock::new();
        MOCK.get_or_init(|| {
            let mock = Arc::new(MockPlatform {
                created: parking_lot::Mutex::new(Vec::new()),
                uploaded: parking_lot::Mutex::new(Vec::new()),
                removed: parking_lot::Mutex::new(Vec::new()),
                fail_upload: AtomicBool::new(false),
            });
            super::set_texture_array_platform(mock.clone());
            mock
        })
    }

    fn fresh_mock() -> (parking_lot::MutexGuard<'static, ()>, Arc<MockPlatform>) {
        let lock = MOCK_LOCK.lock();
        let mock = shared_mock();
        mock.created.lock().clear();
        mock.uploaded.lock().clear();
        mock.removed.lock().clear();
        mock.fail_upload.store(false, Ordering::SeqCst);
        (lock, mock.clone())
    }

    fn canvas(width: i32, height: i32) -> Size<DevicePixels> {
        size(DevicePixels(width), DevicePixels(height))
    }

    fn blocks(width: i32, height: i32) -> Vec<u8> {
        vec![0u8; TextureArrayFormat::Bc7RgbaUnorm.layer_len(canvas(width, height)).unwrap()]
    }

    #[test]
    fn create_reports_missing_platform() {
        let error = require_platform(None)
            .err()
            .expect("a missing backend is an error");
        assert_eq!(
            error.to_string(),
            "texture arrays are not supported on this platform"
        );
    }

    #[test]
    fn create_rejects_zero_layers_and_zero_size() {
        let (_lock, _mock) = fresh_mock();
        assert!(create(canvas(384, 216), 0, TextureArrayFormat::Bc7RgbaUnorm).is_err());
        assert!(create(canvas(0, 216), 1, TextureArrayFormat::Bc7RgbaUnorm).is_err());
    }

    #[test]
    fn create_requires_dimensions_divisible_by_four() {
        let (_lock, _mock) = fresh_mock();
        assert!(create(canvas(383, 216), 1, TextureArrayFormat::Bc7RgbaUnorm).is_err());
        assert!(create(canvas(384, 216), 1, TextureArrayFormat::Bc7RgbaUnorm).is_ok());
    }

    #[test]
    fn reserved_atlas_id_round_trips() {
        let id = TextureArrayId(7);
        let atlas = id.atlas_texture_id();
        assert_eq!(atlas.index, TEXTURE_ARRAY_INDEX_BASE + 7);
        assert_eq!(atlas.kind, AtlasTextureKind::Polychrome);
        assert_eq!(TextureArrayId::from_atlas_texture_id(atlas), Some(id));
    }

    #[test]
    fn bc7_layer_length_is_one_byte_per_pixel() {
        assert_eq!(
            TextureArrayFormat::Bc7RgbaUnorm.layer_len(canvas(384, 216)),
            Some(384 * 216)
        );
        assert_eq!(
            TextureArrayFormat::Bc7RgbaUnorm.layer_len(canvas(256, 144)),
            Some(256 * 144)
        );
    }

    #[test]
    fn upload_validates_layer_and_length() {
        let (_lock, mock) = fresh_mock();
        let array = create(canvas(8, 8), 2, TextureArrayFormat::Bc7RgbaUnorm).expect("created");

        assert!(upload(array, 2, &blocks(8, 8)).is_err());
        assert!(upload(array, 0, &blocks(8, 8)[..63]).is_err());
        upload(array, 0, &blocks(8, 8)).expect("uploaded");
        assert_eq!(mock.uploaded.lock().len(), 1);
    }

    #[test]
    fn upload_advances_only_the_target_layer_version() {
        let (_lock, _mock) = fresh_mock();
        let array = create(canvas(8, 8), 2, TextureArrayFormat::Bc7RgbaUnorm).expect("created");
        let before = live_state(array.id()).expect("state");
        let versions_before: Vec<u32> = before
            .versions
            .iter()
            .map(|version| version.load(Ordering::Relaxed))
            .collect();

        upload(array, 1, &blocks(8, 8)).expect("uploaded");

        let after = live_state(array.id()).expect("state");
        let versions_after: Vec<u32> = after
            .versions
            .iter()
            .map(|version| version.load(Ordering::Relaxed))
            .collect();
        assert_eq!(versions_before[0], versions_after[0]);
        assert!(versions_after[1] > versions_before[1]);
    }

    #[test]
    fn a_failed_upload_does_not_advance_the_version() {
        let (_lock, mock) = fresh_mock();
        let array = create(canvas(8, 8), 1, TextureArrayFormat::Bc7RgbaUnorm).expect("created");
        let before = live_state(array.id()).expect("state").versions[0].load(Ordering::Relaxed);
        mock.fail_upload.store(true, Ordering::SeqCst);

        assert!(upload(array, 0, &blocks(8, 8)).is_err());

        let after = live_state(array.id()).expect("state").versions[0].load(Ordering::Relaxed);
        assert_eq!(before, after);
    }

    #[test]
    fn remove_releases_the_array_once() {
        let (_lock, mock) = fresh_mock();
        let array = create(canvas(8, 8), 1, TextureArrayFormat::Bc7RgbaUnorm).expect("created");

        super::remove(array);
        super::remove(array);

        assert!(live_state(array.id()).is_err());
        assert_eq!(mock.removed.lock().as_slice(), &[array.id()]);
    }

    #[test]
    fn sub_tile_maps_the_visible_fraction_to_texture_pixels() {
        let canvas_size = canvas(400, 200);
        let image = Bounds {
            origin: point(Pixels(-50.), Pixels(0.)),
            size: size(Pixels(400.), Pixels(200.)),
        };
        let visible = Bounds {
            origin: point(Pixels(0.), Pixels(0.)),
            size: size(Pixels(300.), Pixels(200.)),
        };

        let tile = super::sub_tile(canvas_size, image, visible);

        assert_eq!(tile.origin.x.0, 50);
        assert_eq!(tile.origin.y.0, 0);
        assert_eq!(tile.size.width.0, 300);
        assert_eq!(tile.size.height.0, 200);
    }

    #[test]
    fn uv_values_are_stable_for_the_renderer() {
        assert_eq!(TextureArrayUv::Normal as u32, 0);
        assert_eq!(TextureArrayUv::Rotate90 as u32, 1);
    }

    #[test]
    fn live_arrays_is_empty_at_start_or_after_removal() {
        let (_lock, _mock) = fresh_mock();
        let array = create(canvas(8, 8), 1, TextureArrayFormat::Bc7RgbaUnorm).expect("created");
        assert!(live_arrays().contains_key(&array.id()));
        array.remove();
        assert!(!live_arrays().contains_key(&array.id()));
    }

    #[test]
    fn array_state_is_shared_by_handle_copies() {
        let (_lock, _mock) = fresh_mock();
        let array = create(canvas(8, 8), 1, TextureArrayFormat::Bc7RgbaUnorm).expect("created");
        let copy = TextureArray { ..array };
        assert_eq!(copy.id(), array.id());
        assert_eq!(live_state(copy.id()).expect("state").layers, 1);
        copy.remove();
        assert!(live_state(array.id()).is_err());
    }

    #[test]
    fn state_versions_start_at_one() {
        let (_lock, _mock) = fresh_mock();
        let array = create(canvas(8, 8), 2, TextureArrayFormat::Bc7RgbaUnorm).expect("created");
        let state: Arc<ArrayState> = live_state(array.id()).expect("state");
        assert!(state.versions.iter().all(|v| v.load(Ordering::Relaxed) == 1));
    }
}
