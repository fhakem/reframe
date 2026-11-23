# Reframe

Conversion de vidéos 16:9 / 2.39:1 vers 9:16 (TikTok, Reels, Shorts) : recadrage intelligent suivi dans le temps, complété par du seam carving temporellement cohérent.

## État

| Jalon | Contenu | Modules |
|---|---|---|
| M0 | I/O ffmpeg (rotation, fps exact, audio copié, BT.709), CLI | `core/io/video.py`, `cli.py` |
| M1 | Énergie backward/forward, DP Numba + torch, masques, insertion, carte d'ordre basse résolution | `core/seams/`, `core/energy/gradient.py` |
| M2 | Énergie sémantique **classique, sans poids** (OpenCV) : saillance, visages/personnes (Haar, HOG), texte (MSER + cartons plats), lignes (FLD/LSD) | `core/energy/{saliency,detect,text,lines,combine}.py` |
| M3 | Détection de plans (histogrammes), flux optique DIS, seams temporellement cohérents (Grundmann), métriques de jitter / flow warping error | `core/shots.py`, `core/flow.py`, `core/seams/temporal.py`, `core/metrics.py` |
| M4 | Trajectoire de crop par plan (stationnaire / pan / suivi, lissage contraint), budget de seams automatique, fallbacks (fond flou, split-screen) | `core/crop/`, `core/fallback.py` |
| Intégration | Pipeline « smart » en deux passes (analyse basse résolution → plan par plan → rendu) | `core/smart.py` |
| M5 | API FastAPI + file de jobs SQLite (un sous-processus par job, annulation, reprise après crash) + front Vite/React | `api/`, `worker/`, `web/` |
| M6 | Boucle de carving fusionnée en place (Numba), décodage directement en basse résolution, rendu parallèle par groupes de plans puis concaténation | `core/seams/fused.py`, `core/smart.py` |

Non livré (demande des téléchargements de poids) : SAM 2, RAFT, TransNetV2, saillance apprise, Depth Anything, et donc l'édition de masque propagée par SAM 2. Les interfaces (`SaliencyEstimator`, `BoxDetector`, `register_flow_estimator`…) sont prêtes pour les brancher. Pas de tests E2E Playwright (téléchargement d'un navigateur).

### Limite principale constatée sur du vrai contenu

Sans masque de sujet appris, les détecteurs classiques ne voient pas un personnage de dessin animé, et un sujet peu texturé (fourrure blanche) coûte presque rien à traverser : les seams peuvent le déformer, surtout quand le terme temporel les maintient en place pendant que le sujet bouge. La saillance classique favorise les textures plutôt que les gros objets, c'est pourquoi elle sert au placement du crop mais est désactivée dans l'énergie des seams par défaut.

## Utilisation

```bash
uv sync --extra api          # ajouter --extra gpu pour torch
uv run reframe input.mp4 -o out.mp4 --preset tiktok --seam-ratio 0.25
uv run reframe input.mp4 -o out.mp4 --mode naive          # baseline M1
uv run reframe-api --data-dir ./reframe-data --port 8000  # API + UI (après le build du front)
cd web && npm install && npm run build                    # front, servi par l'API
```

`--seam-ratio` est la fraction de la largeur **recadrée** retirée par les seams : l'image est d'abord recadrée à `largeur_cible / (1 - ratio)`, puis réduite par seams jusqu'à `largeur_cible`. 0 = crop centré pur ; au-delà d'environ 0.35, les seams traversent les sujets ; 1920 → 608 par seams seuls demande environ 0.68.

Limites connues : sortie toujours en 8 bits BT.709 (pas de préservation HDR / BT.2020) ; encodage à fps constant (`r_frame_rate`), donc dérive possible sur les sources VFR ; une source plus étroite que 9:16 est recadrée en hauteur.

## Tests

```bash
uv run pytest -m "not gpu"              # CI rapide (génère ses clips via ffmpeg lavfi)
uv run pytest -m gpu                    # parité/bench torch sur MPS/CUDA
uv run pytest tests/test_bench.py --benchmark-only
cd web && npm test -- --run
```
