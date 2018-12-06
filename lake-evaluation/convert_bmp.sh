#!/bin/bash

find $1 -name '*.png' | sed -e 's/\.png$//' | xargs -t -I{} convert {}.png {}.bmp
