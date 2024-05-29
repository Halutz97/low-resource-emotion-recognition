To run the FER model
1) Install packages in requirements.txt (in a virtual environment, preferrably!)
2) Change the lines below in the packages

It is better to do it in a virtual environment, so that we don't make changes in the general Python installation -> that can cause problems later on
In a virtual environment everything is contained, and worst case we can delete the environment and create a new.

To create a virtual environment:
In the terminal, run:

python -m venv emo_env

where "emo_env" is the name of the virtual environment, but it can be anything.

To activate the virtual environment on Windows, run:

.\emo_env\Scripts\activate

After activating the first time, VS Code should detect it automatically. 
When inside a Python script, VS Code will show the current environment (if any) in the bottom right hand corner.

Package files to change:
(If a virtual environmenthas been created, then we can simply find the package files in the VS Code file explorer)

===================================================================================================================

File: /usr/local/lib/python3.10/dist-packages/keras_vggface/models.py
from keras.utils import layer_utils ==> from tensorflow.python.keras.utils import layer_utils

File: /usr/local/lib/python3.10/dist-packages/keras_vggface/models.py
from keras.utils.data_utils import get_file ==> from tensorflow.python.keras.utils.data_utils import get_file

File: /usr/local/lib/python3.10/dist-packages/keras_vggface/utils.py
from keras.utils.data_utils import get_file ==> from tensorflow.python.keras.utils.data_utils import get_file

File: /usr/local/lib/python3.10/dist-packages/keras_vggface/models.py
from keras.engine.topology import get_source_inputs ==> from tensorflow.python.keras.utils.layer_utils import get_source_inputs