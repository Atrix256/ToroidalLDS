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


im = Image.open('out/gauss.png')
im.save('out/gauss.pdf')
im = Image.open('out/triangle.png')
im.save('out/triangle.pdf')

df = pd.read_csv('out/gauss.csv').drop(['Samples'], axis=1)
fig = df.plot(loglog=True, xlabel='Samples', ylabel='Abs Error', title='').get_figure()
fig.axes[0].set_xscale('log', base=2)
fig.axes[0].set_yscale('log', base=2)
fig.savefig('out/integration_gauss.png')
fig.savefig('out/integration_gauss.pdf')
plt.close(fig)

df = pd.read_csv('out/triangle.csv').drop(['Samples'], axis=1)
fig = df.plot(loglog=True, xlabel='Samples', ylabel='Abs Error', title='').get_figure()
fig.axes[0].set_xscale('log', base=2)
fig.axes[0].set_yscale('log', base=2)
fig.savefig('out/integration_triangle.png')
fig.savefig('out/integration_triangle.pdf')
plt.close(fig)



watermark_obj = PdfFileReader('out/gauss.pdf')
watermark_page = watermark_obj.getPage(0)
pdf_reader = PdfFileReader('out/integration_gauss.pdf')
pdf_writer = PdfFileWriter()
for page in range(pdf_reader.getNumPages()):
    page = pdf_reader.getPage(page)
    page.mergeTranslatedPage(watermark_page, tx='65', ty='45')
    pdf_writer.addPage(page)

with open("out/integration_gauss.pdf", 'wb') as out:
    pdf_writer.write(out)


watermark_obj = PdfFileReader('out/triangle.pdf')
watermark_page = watermark_obj.getPage(0)
pdf_reader = PdfFileReader('out/integration_triangle.pdf')
pdf_writer = PdfFileWriter()
for page in range(pdf_reader.getNumPages()):
    page = pdf_reader.getPage(page)
    page.mergeTranslatedPage(watermark_page, tx='65', ty='45')
    pdf_writer.addPage(page)

with open("out/integration_triangle.pdf", 'wb') as out:
    pdf_writer.write(out)

#input1 = open('out/integration_gauss.pdf', "rb")
#input2 = open('out/gauss.pdf', "rb")
#merger = PdfFileMerger()
#merger.append(fileobj = input1, pages=(0,1))
#merger.merge(position=0, fileobj=input2, pages=(0,1))
#output = open("out/integration_gauss.pdf", "wb")
#merger.write(output)



#im = Image.open('out/integration_gauss.png')
#fn = Image.open('out/gauss.png')
#w,h = im.size
#draw = ImageDraw.Draw(im)
#draw.rectangle([(95-1, 350-1),(95+64, 350+64)],outline="#000000")
#im.paste(fn, (95, 350))
#im.save('out/integration_gauss.png')

#im = Image.open('out/integration_triangle.png')
#fn = Image.open('out/triangle.png')
#w,h = im.size
#draw = ImageDraw.Draw(im)
#draw.rectangle([(95-1, 350-1),(95+64, 350+64)],outline="#000000")
#im.paste(fn, (95, 350))
#im.save('out/integration_triangle.png')
