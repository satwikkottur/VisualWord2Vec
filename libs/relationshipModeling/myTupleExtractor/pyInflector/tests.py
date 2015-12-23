#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2006 Bermi Ferrer Martinez
#
# bermi a-t bermilabs - com
#
import unittest
from inflector import Inflector, English

conv = Inflector()
print conv.singularize("people")

