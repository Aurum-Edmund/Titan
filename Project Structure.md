titan\_kit/

├─ configs/

│  └─ titan\_100m.yaml

├─ scripts/

│  └─ gen\_echo\_dataset.py

├─ src/titan/

│  ├─ core.py                 # Titan model (GRU+SGN), tied head

│  ├─ train\_titan.py          # minimal trainer (pads labels=-100)

│  ├─ sampler.py              # greedy + structured sampler

│  ├─ math\_core.py            # exact integer math + helpers

│  └─ programming\_core\_py.py  # tiny deterministic Python subset (optional later)

├─ tests/

│  ├─ test\_tokenizer\_specials.py

│  ├─ test\_dataloader\_padding.py

│  ├─ test\_echo\_overfit.py

│  ├─ test\_math\_core.py

│  └─ test\_sampler\_echo.py

├─ tools/

│  ├─ titan\_probe.py

│  └─ check\_kit.py            # runs a battery of sanity checks

├─ RUN.md

└─ MARY\_CHECKLIST.md          # running task list (“Mary” = the list)



