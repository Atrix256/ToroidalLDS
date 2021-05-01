# This is run at the end of the C++ program.
# This uses the csv's in the "out" folder to generate the graph images in the "out" folder.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio

from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

from os import pardir
from os.path import abspath, basename, dirname, join
from sys import argv, path, stderr

from PyPDF2 import PdfFileMerger, PdfFileWriter, PdfFileReader


df = pd.read_csv('out/gauss.csv').drop(['Samples'], axis=1)
fig = df.plot(loglog=True, xlabel='Samples', ylabel='Abs Error', title='').get_figure()
fig.axes[0].set_xscale('log', base=2)
fig.axes[0].set_yscale('log', base=2)
fig.savefig('out/integration_gauss.png')
plt.close(fig)

df = pd.read_csv('out/triangle.csv').drop(['Samples'], axis=1)
fig = df.plot(loglog=True, xlabel='Samples', ylabel='Abs Error', title='').get_figure()
fig.axes[0].set_xscale('log', base=2)
fig.axes[0].set_yscale('log', base=2)
fig.savefig('out/integration_triangle.png')
plt.close(fig)

im = Image.open('out/integration_gauss.png')
fn = Image.open('out/gauss.png')
w,h = im.size
draw = ImageDraw.Draw(im)
draw.rectangle([(95-1, 350-1),(95+64, 350+64)],outline="#000000")
im.paste(fn, (95, 350))
im.save('out/integration_gauss.png')

im = Image.open('out/integration_triangle.png')
fn = Image.open('out/triangle.png')
w,h = im.size
draw = ImageDraw.Draw(im)
draw.rectangle([(95-1, 350-1),(95+64, 350+64)],outline="#000000")
im.paste(fn, (95, 350))
im.save('out/integration_triangle.png')
