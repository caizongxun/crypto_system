# 🚀 Crypto System - 超快速 Colab 開始

## 最簡單的方式（推薦）

### Step 1: 在 Colab 中執行此 Cell

```bash
!pip install -q pandas numpy tensorflow scikit-learn ccxt pyarrow pandas-ta matplotlib
```

**完成時會看到**: 沒有錯誤訊息，或只有版本警告（可忽略）

### Step 2: 執行訓練

```python
import requests, time
url = 'https://raw.githubusercontent.com/caizongxun/crypto_system/main/v1/train_v1_minimal.py?t=' + str(int(time.time()))
exec(requests.get(url).text)
```

**完成時會看到**:
```
================================================================================
✓ TRAINING COMPLETE!
================================================================================
```

---

## 說明

- **!pip install**: 安裝所有必要套件（約 2-3 分鐘）
- **train_v1_minimal.py**: 用合成數據快速測試（約 2-3 分鐘）
  - 生成假數據（無需等待 API）
  - 建立並訓練模型
  - 保存模型

- **之後**: 可執行 `train_v1.py` 用真實 Binance 數據訓練（45-60 分鐘）

---

## 檢查結果

```python
from pathlib import Path
cache_dir = Path('/content/all_models/v1')
for f in cache_dir.glob('*.h5'):
    print(f"{f.name}: {f.stat().st_size / (1024**2):.1f} MB")
```

---

完成！
