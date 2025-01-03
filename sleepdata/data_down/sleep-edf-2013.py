import requests

import os
base_url = "https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/"
files = [
    "SC4001E0-PSG.edf",
    "SC4001EC-Hypnogram.edf",
    "SC4002E0-PSG.edf",
    "SC4002EC-Hypnogram.edf",
    "SC4011E0-PSG.edf",
    "SC4011EH-Hypnogram.edf",
    "SC4012E0-PSG.edf",
    "SC4012EC-Hypnogram.edf",
    "SC4021E0-PSG.edf",
    "SC4021EH-Hypnogram.edf",
    "SC4022E0-PSG.edf",
    "SC4022EJ-Hypnogram.edf",
    "SC4031E0-PSG.edf",
    "SC4031EC-Hypnogram.edf",
    "SC4032E0-PSG.edf",
    "SC4032EP-Hypnogram.edf",
    "SC4041E0-PSG.edf",
    "SC4041EC-Hypnogram.edf",
    "SC4042E0-PSG.edf",
    "SC4042EC-Hypnogram.edf",
    "SC4051E0-PSG.edf",
    "SC4051EC-Hypnogram.edf",
    "SC4052E0-PSG.edf",
    "SC4052EC-Hypnogram.edf",
    "SC4061E0-PSG.edf",
    "SC4061EC-Hypnogram.edf",
    "SC4062E0-PSG.edf",
    "SC4062EC-Hypnogram.edf",
    "SC4071E0-PSG.edf",
    "SC4071EC-Hypnogram.edf",
    "SC4072E0-PSG.edf",
    "SC4072EH-Hypnogram.edf",
    "SC4081E0-PSG.edf",
    "SC4081EC-Hypnogram.edf",
    "SC4082E0-PSG.edf",
    "SC4082EP-Hypnogram.edf",
    "SC4091E0-PSG.edf",
    "SC4091EC-Hypnogram.edf",
    "SC4092E0-PSG.edf",
    "SC4092EC-Hypnogram.edf",
    "SC4101E0-PSG.edf",
    "SC4101EC-Hypnogram.edf",
    "SC4102E0-PSG.edf",
    "SC4102EC-Hypnogram.edf",
    "SC4111E0-PSG.edf",
    "SC4111EC-Hypnogram.edf",
    "SC4112E0-PSG.edf",
    "SC4112EC-Hypnogram.edf",
    "SC4121E0-PSG.edf",
    "SC4121EC-Hypnogram.edf",
    "SC4122E0-PSG.edf",
    "SC4122EV-Hypnogram.edf",
    "SC4131E0-PSG.edf",
    "SC4131EC-Hypnogram.edf",
    "SC4141E0-PSG.edf",
    "SC4141EU-Hypnogram.edf",
    "SC4142E0-PSG.edf",
    "SC4142EU-Hypnogram.edf",
    "SC4151E0-PSG.edf",
    "SC4151EC-Hypnogram.edf",
    "SC4152E0-PSG.edf",
    "SC4152EC-Hypnogram.edf",
    "SC4161E0-PSG.edf",
    "SC4161EC-Hypnogram.edf",
    "SC4162E0-PSG.edf",
    "SC4162EC-Hypnogram.edf",
    "SC4171E0-PSG.edf",
    "SC4171EU-Hypnogram.edf",
    "SC4172E0-PSG.edf",
    "SC4172EC-Hypnogram.edf",
    "SC4181E0-PSG.edf",
    "SC4181EC-Hypnogram.edf",
    "SC4182E0-PSG.edf",
    "SC4182EC-Hypnogram.edf",
    "SC4191E0-PSG.edf",
    "SC4191EP-Hypnogram.edf",
    "SC4192E0-PSG.edf",
    "SC4192EV-Hypnogram.edf"
]

for file in files:
    url = base_url + file
    #下载文件
    r = requests.get(url, allow_redirects=True)
    #保存文件
    open(str(file), 'wb').write(r.content)