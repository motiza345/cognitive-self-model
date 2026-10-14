# cognitive-layer v0.2

یک لایهٔ خودمدل بیرونی، قابل ممیزی و مبتنی بر شواهد برای عامل‌های LLM.

در نسخهٔ ۰٫۲، سیگنال اصلی **اطمینان خود مدل در هر نمونه** است، مثلاً احتمال پاسخ از روی logprob. ادعاها هم به **کالیبراسیون محدوده‌دار** تبدیل شده‌اند: «در محدودهٔ S، اطمینان من دربارهٔ درستی پاسخم آگاهی‌بخش است یا نیست، و نگاشت اطمینان به احتمال این است».

## چرا این تغییر

دو دلیل از گزارش فنی نسخهٔ ۱٫۱ آمده است:

- **سیگنال قوی‌تر:** در M34a-v2، احتمال پاسخ خود مدل +۰٫۰۸۹ سود داد (AUROC برابر ۰٫۹۳)، اما سابقهٔ عملکرد در هر سطح فقط +۰٫۰۲۵.
- **کنترل بیرونی:** پژوهش‌های RiskEval و MIRROR نشان می‌دهند مدل‌ها اطمینانشان را به تصمیم تبدیل نمی‌کنند، اما یک کنترل بیرونی این کار را می‌کند.

## اجرا

```bash
pip install numpy scipy pytest
PYTHONPATH=. python -m pytest -q          # 15 آزمون
PYTHONPATH=. python scripts/sensitivity.py
```

## استفاده

```python
from cognitive_layer import (CalibrationRegistry, CalibrationRule, CognitiveLayer, ExpectedUtilityPolicy,
                             UtilityTable, Ledger, Scope, from_token_logprobs)

reg = CalibrationRegistry(CalibrationRule(min_each=15))           # GLOBAL به‌صورت خودکار
reg.declare("sql_prod", "My confidence is informative on prod SQL", Scope.of(task="sql", db="prod"))
layer = CognitiveLayer(reg, ExpectedUtilityPolicy(UtilityTable(verify=0.7), epsilon=0.05),
                       ledger=Ledger("self_ledger.jsonl"))

conf = from_token_logprobs(answer_token_logprobs)                 # تعریف M34a
pred = layer.before({"task": "sql", "db": "prod"}, confidence=conf)
# pred.action ∈ {ACT, VERIFY, ABSTAIN, DELEGATE}
layer.after(pred.prediction_id, success=was_own_answer_correct_or_None, provenance="unit_test:v3")
```

ساختار فایل‌ها: `core` · `calibration` · `registry` · `policy` · `sensors` · `layer` (+`replay`) · `ledger` · `evaluation` · `sim`
