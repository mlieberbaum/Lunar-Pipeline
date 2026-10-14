# Lunar pipeline v5: changes from v4.1

Entry point, MATLAB wrapper and pinned dependencies are unchanged:
`build_texture(orbiter, seamless, morning, evening, output, source_workers=1)`.
Processing remains deterministic and conventional (filters, resampling,
registration, masks); no generative or trained models.

## Why

v4.1 stretched each Kaguya product to its own 0.5/99.5 percentiles and then
judged pixel validity on the stretched values (`.01 < v < .995`). Valid but
dark crater floors and bright walls were declared unusable and sent through
synthetic quilting and several repair-of-repair stages: the crater-bottom
splotches and "obviously repaired" patches. The same stretch made detail
contrast depend on each tile's histogram (seams between subtiles).

## What changed

All source handling goes through `lunar_source.py` and `lunar_geometry.py`.

1. **Validity from the archive labels.** Sources are stored as codes linear
   in label DN (`2*DN`). DUMMY (0), low saturation (1) and high saturation
   (32767) become 0. A sample is usable exactly when its code is non-zero.
2. **One detail definition, fixed gains.** Detail is multiplicative detail
   of log reflectance about a local mean: mid scales (2-32 px) gain 1.5,
   fine scales (<2 px) gain 2.2. No clipping plateaus, no unsharp-mask
   halos. Morning/Evening are scaled to Seamless contrast measured where
   both are valid, and contribute texture only (<= 8 px) so low-sun
   shading cannot print as streaks.
3. **LROC-anchored local gain.** Kaguya relative contrast is compared with
   the Orbiter reference in the band both resolve, over ~3.5 km windows,
   and detail amplitude follows the reference (mare unchanged, highland
   strips with harsh relief toned down ~25%; never amplified). Detail also
   responds to reference brightness: bright crater walls and floors get
   proportionally less (x0.73 at tone .6, x0.48 at 1.0, like v4.1's
   `(v-m)/(m+.15)`), and deep reference shadows keep a smooth tone with
   faint texture (x0.35). This keeps large craters calm instead of speckled.
4. **Geometry.** Seamless mosaic strips carry along-track errors of up to
   ~20 px varying across each strip, with steps at seams (tears). Morning
   (one orbit, consistent with the reference) now defines the output grid;
   Seamless is warped into it with a displacement field measured on a
   512 px grid and filtered only within each strip; Evening gets its own
   field. The Orbiter reference is resampled onto the anchor grid instead of
   shifting Kaguya, so tile edges keep real data. Falls back to the Seamless
   grid if Morning cannot be registered.
5. **Seams: v4.1 behaviour restored.** Seamless is excluded around each
   detected seam with v4.1's wide feathered masks (about +-170 px plus half
   the seam width, fading over 160 px) and the repair stages use v4.1's
   narrower masks (+-32 px, 64 px fade). Morning, then Evening, fill the band
   at full detail scale, so neighbouring strips blend gradually. Seams are
   still traced row by row, but only to group Seamless strips for the
   geometry warp. Two changes to the fill: Evening is aligned against warped
   Seamless (Morning is often dark exactly where Evening is needed), and
   Evening fill detail is halved (`EVENING_FILL_SCALE = .5`) because its
   opposite-sun relief, used with reversed polarity, otherwise prints bright
   dots.
6. **Shadows and penumbra.** Log detail is smoothly limited (tanh, about
   x0.37..x2.7), so near-black penumbra samples cannot print dark slots.
   Where no product has data and the reference is dark (permanent shadow),
   the reference tone is kept instead of synthetic texture.
7. **Resampling.** Cubic-spline sampling replaces bicubic resize plus
   bilinear shift; reduces an 8.26 px beat grid (3600 vs 4096 px/deg).
8. Registration and strip detection still run on a stretched preview; all
   repair stages remain as a safety net and now act on far fewer pixels.

## Inputs

Use the original `.img` files. PNG inputs must contain unmodified label DN,
not stretched previews.

## Diagnostics

`*.diagnostics.json` adds `geometry` (anchor, seam traces, field ranges),
`detail_gain_field_percentiles_5_50_95`, per-source `detail_gain`.
Set `LUNAR_DEBUG_DIR` to keep intermediate masks.

## Results on the sample tiles (pixels changed by repair stages)

| tile | v4.1 | v5 |
|------|------|----|
| 217h | 5.3% | 0.01% |
| 407a | 5.4% | 0.74% |
| 407g | 9.7% | 2.4% |

217h is 99.5% byte-identical to the previous v5 checkpoint (it has no seams).

## Known open items

- Seam bands use Morning/Evening (low-sun) texture and, where both are dark,
  quilted texture, as in v4.1; at full resolution a band can show low-sun
  relief or borrowed texture, but it does not read as a stripe.
- Small no-data spots can be filled as smooth patches by the tone solve.
- Mosaic patches inside Seamless that are not north-south seams (e.g. a
  higher-sun patch on a crater floor) can leave a faint outline.
- Neighbor-tile boundaries (aprons) not yet addressed.
