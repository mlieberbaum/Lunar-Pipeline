Lunar texture pipeline v4.1

Extract into your existing Python source folder. Install requirements.txt in the Python environment MATLAB uses, then restart MATLAB. Keep sourceWorkers=1 for parfor.

Entry point and argument order are unchanged:
  from lunar_texture_auto import build_texture
  build_texture(orbiter, seamless, morning, evening, output, source_workers=1)

Orbiter is uint8 grayscale/RGB PNG, any dimensions. Seamless is 10800-square uint16 PNG or original big-endian IMG. Morning/Evening are 12288-square uint16 PNG or original IMG. IMG receives the existing 0.5/99.5 percentile stretch; PNG is not stretched again. Final output is 12288-square lossless uint16 PNG plus JSON diagnostics. Existing output files are rejected. Originals are never modified.

Conventional, fully local processing: automatic registration and validity masks; measured-source recovery; fixed-seed raster quilting from a sorted whole-tile donor pool; tiny-hole interpolation; whole-tile residual detection; frequency and tone blending through map-derived repair perimeters. No generative AI, trained model, tile IDs, or hand-picked geographic repair coordinates. Exemplar placement introduces synthetic texture, not recovered geography.

One source worker executes synchronously. BLAS and broad-kernel FFT processing are limited to one compute thread. Multiple MATLAB workers still multiply memory use and temporary disk I/O. Every build_texture call creates its own random lunar_work_* folder beside the output. All intermediate filenames are confined to that folder. Memory mappings are explicitly closed before cleanup, including on exceptions, so Windows can delete the files. Different MATLAB parfor workers must use different output filenames; duplicate output claims fail immediately. The output lock is removed automatically after success or failure. Use fast local storage with ample free space.

Deterministic repeatability requires identical input pixels, parameters, dependency versions and processing extent. PNG decoding preserves the uint16 output pixels exactly. Cross-platform floating-point bit identity is not promised.

Diagnostics include fallback coverage, repair assignment, protected-pixel checks and timings. Every detector-identified defect has a repair route, including tiny defects. Additional tiles still require visual review; this is not a guarantee that every possible artifact will be detected. Neighbor-to-neighbor geographic boundary alignment is a separate task.

This ZIP contains runtime files only. Benchmark and comparison files are supplied separately.

Install dependencies using the Python executable configured in MATLAB:
  python -m pip install -r requirements.txt
Then restart MATLAB to discard previously imported Python modules. The existing MATLAB wrapper and sourceWorkers argument are unchanged. Tested with Python 3.12.

The approximately 15-minute target is a processing benchmark on the supplied reference inputs, not a fixed runtime for every tile or computer. Concurrent MATLAB workers compete for memory and disk bandwidth; benchmark one tile with sourceWorkers=1 on your machine before increasing parfor concurrency.

V4.1 freezes the v4 image algorithms, thresholds, donor ordering, interpolation, repairs and blending. Runtime-only changes: exact faster mask operations, cached FFT kernel transforms with smaller batches, reused source mappings and less PNG compression. PNG compressed bytes can differ; decoded uint16 pixels are the regression target. Dependencies remain pinned.

Successful completion leaves only the final PNG and diagnostics JSON. Normal processing exceptions and KeyboardInterrupt also remove the per-call workspace and lock. If the process is forcibly killed or the machine loses power, Python cannot execute cleanup; after all workers stop, abandoned lunar_work_* folders and matching .lunar.lock files can be removed. Never delete working folders while workers are running.
