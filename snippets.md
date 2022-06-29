
Ignore warnings
```python
from pandas.core.common import SettingWithCopyWarning
from matplotlib_inline import backend_inline

import warnings

backend_inline.set_matplotlib_formats('svg')
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
```
