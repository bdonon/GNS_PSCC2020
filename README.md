# GNS_PSCC2020

Code that was used in the development of the paper https://hal.archives-ouvertes.fr/hal-02372741v1

Disclaimer : This code is not maintained, as a more generic version of this work is on the way. 
However, some parts of this code could still be useful to some, so I still decided to put it online.

## Create virtual environment & activate it
``
virtualenv ENV -p python3
source ENV/bin/activate
``

## Install requirements
``
pip install -r requirements
``

## Start a training
``
python train --case case14
``
## License information

Copyright 2019-2020 RTE and INRIA (France)

RTE: http://www.rte-france.com
INRIA: https://www.inria.fr/
This Source Code is subject to the terms of the GNU Lesser General Public License v3.0. If a copy of the LGPL-v3 was not distributed with this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.fr.html.
