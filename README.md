# Reframe

Conversion de vidéos 16:9 / 2.39:1 vers 9:16 (TikTok, Reels, Shorts) : recadrage intelligent suivi dans le temps, complété par du seam carving temporellement cohérent.

Pour l'instant, seuls le squelette du paquet et l'outillage (ruff, mypy, pytest) sont en place.

## Plan

| Jalon | Contenu |
|---|---|
| M0 | I/O ffmpeg (rotation, fps exact, audio copié, BT.709), CLI |
| M1 | Énergie backward/forward, DP Numba + torch, masques, insertion, carte d'ordre |
| M2 | Énergie sémantique classique, sans poids |
| M3 | Plans, flux optique, seams temporellement cohérents |
| M4 | Trajectoire de crop par plan, budget de seams, fallbacks |
| M5 | API FastAPI, file de jobs, front |
| M6 | Performance |

## Développement

```bash
uv sync
uv run pytest
```
