from uuid import uuid1
from datetime import datetime
import os
import time

def get_version():
    model_id = str(uuid1())
    date = datetime.now().date().strftime("%Y_%m_%d")
    version = f'{date}/{model_id}'
    return version