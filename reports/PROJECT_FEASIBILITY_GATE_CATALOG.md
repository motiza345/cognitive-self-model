# Feasibility-gate catalog

This catalog is the preregistration required by `reports/PROJECT_FEASIBILITY_GATE.md` section 9. It was written before any feasibility-gate forward. The gate criteria are unchanged.

Ids use the prefix `f-`. Within each surface family, index `k` starting at 1 uses `(k - 1) mod 4`: train, validation, evaluation, replication. Each partition has 12 prompts, four from each family. Family is a balance label. It is not a model input.

| prompt_id | family | partition | text |
| --- | --- | --- | --- |
| `f-completion-01` | completion | train | The boiling point of ethanol is |
| `f-completion-02` | completion | validation | A standard chessboard has |
| `f-completion-03` | completion | evaluation | The painter of the Mona Lisa is |
| `f-completion-04` | completion | replication | Mercury is a metal that is |
| `f-completion-05` | completion | train | The Nile is a |
| `f-completion-06` | completion | validation | A pentagon has five |
| `f-completion-07` | completion | evaluation | The currency of Canada is the |
| `f-completion-08` | completion | replication | Glass is usually melted in a |
| `f-completion-09` | completion | train | The slowest mammal is the |
| `f-completion-10` | completion | validation | An hour has sixty |
| `f-completion-11` | completion | evaluation | The chemical symbol for silver is |
| `f-completion-12` | completion | replication | Cotton grows on a |
| `f-completion-13` | completion | train | The inventor of the telephone is |
| `f-completion-14` | completion | validation | A leap year has |
| `f-completion-15` | completion | evaluation | The largest bone in the body is the |
| `f-completion-16` | completion | replication | Honey is made by a |
| `f-syntax-01` | syntax | train | After the bridge closed, |
| `f-syntax-02` | syntax | validation | If the ink spills, |
| `f-syntax-03` | syntax | evaluation | Before the curtain fell, |
| `f-syntax-04` | syntax | replication | Although the well was dry, |
| `f-syntax-05` | syntax | train | Because the rope had snapped, |
| `f-syntax-06` | syntax | validation | Unless the gate is shut, |
| `f-syntax-07` | syntax | evaluation | While the dough was rising, |
| `f-syntax-08` | syntax | replication | As soon as the whistle blew, |
| `f-syntax-09` | syntax | train | Even though the lens was cracked, |
| `f-syntax-10` | syntax | validation | Whenever the tide turns, |
| `f-syntax-11` | syntax | evaluation | Since the harbor was empty, |
| `f-syntax-12` | syntax | replication | Until the frost ended, |
| `f-syntax-13` | syntax | train | Once the letter was sealed, |
| `f-syntax-14` | syntax | validation | Provided the axle is greased, |
| `f-syntax-15` | syntax | evaluation | Whether or not the lamp works, |
| `f-syntax-16` | syntax | replication | Whereas the second sample passed, |
| `f-instruction-01` | instruction | train | Reply with one word. A gas used in balloons: |
| `f-instruction-02` | instruction | validation | Reply with one word. The meal eaten at dawn: |
| `f-instruction-03` | instruction | evaluation | Name a tool used for digging: |
| `f-instruction-04` | instruction | replication | Name a month with thirty days: |
| `f-instruction-05` | instruction | train | Reply with one word. Opposite of narrow: |
| `f-instruction-06` | instruction | validation | Name a kind of grain: |
| `f-instruction-07` | instruction | evaluation | Reply with one word. A baked fruit dessert: |
| `f-instruction-08` | instruction | replication | Name a percussion instrument: |
| `f-instruction-09` | instruction | train | Reply with one word. The number of sides on a pentagon: |
| `f-instruction-10` | instruction | validation | Name a sea: |
| `f-instruction-11` | instruction | evaluation | Reply with one word. A fish that can shock: |
| `f-instruction-12` | instruction | replication | Name a piece of jewelry: |
| `f-instruction-13` | instruction | train | Reply with one word. A stone used in pencils: |
| `f-instruction-14` | instruction | validation | Name a unit of length: |
| `f-instruction-15` | instruction | evaluation | Reply with one word. The color of emeralds: |
| `f-instruction-16` | instruction | replication | Name a room in a house: |
