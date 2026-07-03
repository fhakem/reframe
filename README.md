# Reframe

Conversion de vidéos 16:9 / 2.39:1 vers 9:16 (TikTok, Reels, Shorts) : recadrage intelligent suivi dans le temps, complété par du seam carving temporellement cohérent.

## État : M0 → M4 + intégration

| Jalon | Contenu | Modules |
|---|---|---|
| M0 | I/O ffmpeg (rotation, fps exact, audio copié, BT.709), CLI | `core/io/video.py`, `cli.py` |
| M1 | Énergie backward/forward, DP Numba + torch, masques, insertion, carte d'ordre basse résolution | `core/seams/`, `core/energy/gradient.py` |
| M2 | Énergie sémantique **classique, sans poids** (OpenCV) : saillance, visages/personnes (Haar, HOG), texte (MSER + cartons plats), lignes (FLD/LSD) | `core/energy/{saliency,detect,text,lines,combine}.py` |
| M3 | Détection de plans (histogrammes), flux optique DIS, seams temporellement cohérents (Grundmann), métriques de jitter / flow warping error | `core/shots.py`, `core/flow.py`, `core/seams/temporal.py`, `core/metrics.py` |
| M4 | Trajectoire de crop par plan (stationnaire / pan / suivi, lissage contraint), budget de seams automatique, fallbacks (fond flou, split-screen) | `core/crop/`, `core/fallback.py` |
| Intégration | Pipeline « smart » en deux passes (analyse basse résolution → plan par plan → rendu) | `core/smart.py` |

Reste : M5 (API, worker, front) et M6 (performance).

## Utilisation

```bash
uv sync --extra gpu                       # torch pour MPS/CUDA
uv run reframe input.mp4 -o out.mp4 --preset tiktok --seam-ratio 0.25
uv run reframe input.mp4 -o out.mp4 --mode naive   # baseline M1
```

`--seam-ratio` est la fraction de la largeur **recadrée** retirée par les seams : l'image est d'abord recadrée à `largeur_cible / (1 - ratio)`, puis réduite par seams jusqu'à `largeur_cible`. 0 = crop centré pur ; au-delà d'environ 0.35, les seams traversent les sujets ; 1920 → 608 par seams seuls demande environ 0.68.

Limites connues : sortie toujours en 8 bits BT.709 (pas de préservation HDR / BT.2020) ; encodage à fps constant (`r_frame_rate`), donc dérive possible sur les sources VFR ; une source plus étroite que 9:16 est recadrée en hauteur.

## Tests

```bash
uv run pytest -m "not gpu"              # CI rapide (génère ses clips via ffmpeg lavfi)
uv run pytest -m gpu                    # parité/bench torch sur MPS/CUDA
uv run pytest tests/test_bench.py --benchmark-only
```
