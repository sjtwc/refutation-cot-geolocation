# Rescored archived results

The old `country_correct` field used naive substring matching. Unparsed responses
were counted as wrong and were never mapped to a ground-truth centroid.

| Run | Images | Old accuracy | Strict accuracy | Coverage | Unparsed | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llava16-base_test1000_2025-12-08 | 1000 | 12.7% | 11.2% | 28.4% | 716 | -1.5 pp |
| llava16-normal_test1000_2025-12-07 | 1000 | 24.7% | 22.3% | 57.1% | 429 | -2.4 pp |
| llava16-normal_test4800_2025-12-07 | 4800 | 30.5% | 32.9% | 91.9% | 389 | +2.4 pp |
| llava16-refutation_test1000_2025-12-07 | 1000 | 32.1% | 34.7% | 93.1% | 69 | +2.6 pp |
