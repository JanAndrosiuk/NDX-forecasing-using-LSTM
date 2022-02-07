#!/usr/bin/env python
# -*- coding: utf-8 -*-
from preprocessing import *

def main():
    icsa_shift()
    technical_indicators()
    concatenate_dfs()
    return 0

if __name__ == "__main__":
    main()