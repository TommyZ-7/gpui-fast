// Polychrome sprites sampled from a BC7 Texture2DArray.
//
// The instance layout is the ordinary PolychromeSprite struct. `tile.tile_id`
// is the array layer, `tile.padding` is the UV rotation flag, and
// `tile.bounds` is the visible sub-rectangle in array pixels.

struct GlobalParams {
    viewport_size: vec2<f32>,
    premultiplied_alpha: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> globals: GlobalParams;
@group(1) @binding(0) var<storage, read> b_poly_sprites: array<PolychromeSprite>;
@group(2) @binding(0) var t_sprite_array: texture_2d_array<f32>;
@group(2) @binding(1) var s_sprite: sampler;

fn load_poly_sprite(instance_id: u32) -> PolychromeSprite {
    return b_poly_sprites[instance_id];
}

const GRAYSCALE_FACTORS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

struct Bounds {
    origin: vec2<f32>,
    size: vec2<f32>,
}

struct Corners {
    top_left: f32,
    top_right: f32,
    bottom_right: f32,
    bottom_left: f32,
}

struct AtlasTextureId {
    index: u32,
    kind: u32,
}

struct AtlasBounds {
    origin: vec2<i32>,
    size: vec2<i32>,
}

struct AtlasTile {
    texture_id: AtlasTextureId,
    tile_id: u32,
    padding: u32,
    bounds: AtlasBounds,
}

struct PolychromeSprite {
    order: u32,
    pad: u32,
    tile: AtlasTile,
    grayscale: u32,
    opacity: f32,
    bounds: Bounds,
    content_mask: Bounds,
    corner_radii: Corners,
}

struct PolySpriteVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) tile_position: vec2<f32>,
    @location(1) @interpolate(flat) sprite_id: u32,
    @location(3) clip_distances: vec4<f32>,
}

fn to_device_position(unit_vertex: vec2<f32>, bounds: Bounds) -> vec4<f32> {
    let position = unit_vertex * vec2<f32>(bounds.size) + bounds.origin;
    let device_position = position / globals.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    return vec4<f32>(device_position, 0.0, 1.0);
}

fn to_tile_position(unit_vertex: vec2<f32>, tile: AtlasTile) -> vec2<f32> {
    let array_size = vec2<f32>(textureDimensions(t_sprite_array, 0));
    // `tile.bounds` is always in display-space pixels. A rotated layer swaps
    // the stored axes, so its display size is the storage array size reversed.
    let rotated = tile.padding == 1u;
    let display_size = select(array_size, array_size.yx, rotated);
    return (vec2<f32>(tile.bounds.origin) + unit_vertex * vec2<f32>(tile.bounds.size)) / display_size;
}

fn distance_from_clip_rect(unit_vertex: vec2<f32>, bounds: Bounds, clip_bounds: Bounds) -> vec4<f32> {
    let position = unit_vertex * vec2<f32>(bounds.size) + bounds.origin;
    let tl = position - clip_bounds.origin;
    let br = clip_bounds.origin + clip_bounds.size - position;
    return vec4<f32>(tl.x, br.x, tl.y, br.y);
}

fn pick_corner_radius(center_to_point: vec2<f32>, corner_radii: Corners) -> f32 {
    let radius = select(
        select(corner_radii.top_left, corner_radii.top_right, center_to_point.x > 0.0),
        select(corner_radii.bottom_left, corner_radii.bottom_right, center_to_point.x > 0.0),
        center_to_point.y > 0.0,
    );
    return max(radius, 0.0);
}

fn quad_sdf_impl(corner_center_to_point: vec2<f32>, corner_radius: f32) -> f32 {
    if (corner_radius == 0.0) {
        return max(corner_center_to_point.x, corner_center_to_point.y);
    }
    let signed_distance_to_inset_quad =
        length(max(vec2<f32>(0.0), corner_center_to_point)) +
        min(0.0, max(corner_center_to_point.x, corner_center_to_point.y));
    return signed_distance_to_inset_quad - corner_radius;
}

fn quad_sdf(point: vec2<f32>, bounds: Bounds, corner_radii: Corners) -> f32 {
    let half_size = bounds.size / 2.0;
    let center = bounds.origin + half_size;
    let center_to_point = point - center;
    let corner_radius = pick_corner_radius(center_to_point, corner_radii);
    let corner_to_point = abs(center_to_point) - half_size;
    let corner_center_to_point = corner_to_point + corner_radius;
    return quad_sdf_impl(corner_center_to_point, corner_radius);
}

fn blend_color(color: vec4<f32>, alpha_factor: f32) -> vec4<f32> {
    let alpha = color.a * alpha_factor;
    let multiplier = select(1.0, alpha, globals.premultiplied_alpha != 0u);
    return vec4<f32>(color.rgb * multiplier, alpha);
}

fn rotate_uv(uv: vec2<f32>, rotation: u32) -> vec2<f32> {
    if (rotation == 1u) {
        // The stored layer was rotated 90° counter-clockwise before upload,
        // so its inverse maps display (u, v) to stored (v, 1 - u).
        return vec2<f32>(uv.y, 1.0 - uv.x);
    }
    return uv;
}

@vertex
fn vs_poly_array_sprite(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> PolySpriteVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let sprite = load_poly_sprite(instance_id);

    var out = PolySpriteVarying();
    out.position = to_device_position(unit_vertex, sprite.bounds);
    out.tile_position = to_tile_position(unit_vertex, sprite.tile);
    out.sprite_id = instance_id;
    out.clip_distances = distance_from_clip_rect(unit_vertex, sprite.bounds, sprite.content_mask);
    return out;
}

@fragment
fn fs_poly_array_sprite(input: PolySpriteVarying) -> @location(0) vec4<f32> {
    let sprite = load_poly_sprite(input.sprite_id);
    let uv = rotate_uv(input.tile_position, sprite.tile.padding);
    let sample = textureSample(t_sprite_array, s_sprite, uv, i32(sprite.tile.tile_id));

    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    let distance = quad_sdf(input.position.xy, sprite.bounds, sprite.corner_radii);
    var color = sample;
    if (sprite.grayscale != 0u) {
        let grayscale = dot(color.rgb, GRAYSCALE_FACTORS);
        color = vec4<f32>(vec3<f32>(grayscale), sample.a);
    }
    return blend_color(color, sprite.opacity * saturate(0.5 - distance));
}
