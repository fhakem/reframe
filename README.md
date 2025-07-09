# Reframe

Conversion de vidéos 16:9 / 2.39:1 vers 9:16 (TikTok, Reels, Shorts) : recadrage intelligent suivi dans le temps, complété par du seam carving temporellement cohérent.

## État : M0 + M1 (socle et seam carving image)

| Fait | Module |
|---|---|
| I/O vidéo via ffmpeg : rotation appliquée, fps exact (`Fraction`), audio copié sans ré-encodage si compatible MP4, sortie H.264 yuv420p BT.709 | `core/io/video.py` |
| Énergie backward (gradient) + forward energy (Rubinstein 2008) | `core/energy/gradient.py` |
| DP générique Numba (CPU) + torch (CUDA/MPS/CPU) | `core/seams/dp.py`, `dp_torch.py` |
| Retrait de seams, masques protect/remove, hook `extra_energy`, insertion | `core/seams/carve.py` |
| Carte d'ordre de retrait calculée en basse résolution puis appliquée en pleine résolution (top-K par ligne) | `core/seams/apply.py` |
| CLI + pipeline **baseline naïve** : crop centré, puis seams frame par frame | `cli.py`, `core/pipeline.py` |

La baseline n'a ni suivi de sujet ni cohérence temporelle : les seams « tremblent » d'une frame à l'autre. Elle sert de référence de jitter pour M3.

Reste à faire : M2 (saillance, visages, SAM 2, lignes), M3 (plans, flux optique, terme de Grundmann), M4 (trajectoire de crop, fallbacks), M5 (API, worker, front), M6 (performance).

## Utilisation

```bash
uv sync                      # ajouter --extra gpu pour torch
uv run reframe input.mp4 -o out.mp4 --preset tiktok --seam-ratio 0.25
```

`--seam-ratio` est la fraction de la largeur **recadrée** retirée par les seams : l'image est d'abord recadrée à `largeur_cible / (1 - ratio)`, puis réduite par seams jusqu'à `largeur_cible`. 0 = crop centré pur ; au-delà d'environ 0.35, les seams traversent les sujets ; 1920 → 608 par seams seuls demande environ 0.68.

Limites connues : sortie toujours en 8 bits BT.709 (pas de préservation HDR / BT.2020) ; encodage à fps constant (`r_frame_rate`), donc dérive possible sur les sources VFR ; une source plus étroite que 9:16 est recadrée en hauteur.

## Tests

```bash
uv run pytest -m "not gpu"              # CI rapide (génère ses clips via ffmpeg lavfi)
uv run pytest -m gpu                    # parité/bench torch sur MPS/CUDA
uv run pytest tests/test_bench.py --benchmark-only
```
