#!/usr/bin/python

import sys, math
import gtk, glib
from common import *

class PoorPlotter(gtk.DrawingArea):

    def __init__(self):
        super(PoorPlotter, self).__init__()
        self.connect("expose_event", self.expose)
        self.set_size_request(900,128)

        self._margins = (10.5,15.5,20.5,30.5)
        self._xtics = tuple(xrange(0,7001,1000))
        self._ytics = tuple(xrange(-10,-71,-10))
        self._xrange = (0,7000)
        self._yrange = (-10,-70)
        self._ticksize = 5
        self._labeloffs = (5,5)
        self._dotradius = 3

        self._width = self._height = 1

        self._mel_data = ()
        self._freq_data = ()
        self._label = None

    def set_data(self, mel_data=None, freq_data=None, label=None):
        if mel_data is not None:
            self._mel_data = mel_data
        if freq_data is not None:
            self._freq_data = freq_data
        if label is not None:
            self._label = label

    def pt2xy(self, pt):
        xspan = self._xrange[1] - self._xrange[0]
        yspan = self._yrange[1] - self._yrange[0]
        w = self._width - self._margins[1] - self._margins[3]
        h = self._height - self._margins[0] - self._margins[2]

        return (1. * (pt[0] - self._xrange[0]) / xspan * w + self._margins[3],
                1. * (pt[1] - self._yrange[0]) / yspan * h + self._margins[0])

    def expose(self, widget, event):
        cr = widget.window.cairo_create()

        cr.rectangle(event.area.x, event.area.y,
                     event.area.width, event.area.height)
        cr.clip()

        self._width, self._height = self.window.get_size()

        cr.set_source_rgb(1., 1., 1.)
        cr.rectangle(0, 0, self._width, self._height)
        cr.fill()

        cr.set_source_rgb(0., 0., 0.)
        cr.set_line_width(1.)
 
        cr.move_to( *self.pt2xy((self._xrange[0], self._yrange[0])) )
        cr.line_to( *self.pt2xy((self._xrange[1], self._yrange[0])) )
        cr.line_to( *self.pt2xy((self._xrange[1], self._yrange[1])) )
        cr.line_to( *self.pt2xy((self._xrange[0], self._yrange[1])) )
        cr.line_to( *self.pt2xy((self._xrange[0], self._yrange[0])) )

        for x in self._xtics:
            px,py = self.pt2xy((x, self._yrange[1]))
            cr.move_to(px,py)
            cr.line_to(px,py-self._ticksize)
            
            px,py = self.pt2xy((x, self._yrange[0]))
            cr.move_to(px,py)
            cr.line_to(px,py+self._ticksize)
        
        for y in self._ytics:
            px,py = self.pt2xy((self._xrange[0], y))
            cr.move_to(px,py)
            cr.line_to(px+self._ticksize,py)
            
            px,py = self.pt2xy((self._xrange[1], y))
            cr.move_to(px,py)
            cr.line_to(px-self._ticksize, py)
        
        cr.stroke()

        for x in self._xtics:
            label = str(x)
            extents = cr.text_extents(label)

            px,py = self.pt2xy((x, self._yrange[1]))
            cr.move_to(px - .5 * extents[2], py + self._labeloffs[1] + extents[3])
            cr.show_text(label)
        
        for y in self._ytics:
            label = str(y)
            extents = cr.text_extents(label)

            px,py = self.pt2xy((self._xrange[0], y))
            cr.move_to(px - self._labeloffs[0] - extents[2], py + 0.5*extents[3])
            cr.show_text(label)

        if self._label is not None:
            extents = cr.text_extents(self._label)

            px,py = self.pt2xy((self._xrange[1], self._yrange[0]))
            cr.move_to(px - self._labeloffs[0] - extents[2], py + self._labeloffs[1] + extents[3])
            cr.show_text(self._label)

        def plot_line(data):
            for idx,(x,y) in enumerate(data):
                px,py = self.pt2xy((x,y))
                if 0 == idx:
                    cr.move_to(px,py)
                else:
                    cr.line_to(px,py)

        cr.set_source_rgba(0., 0., 0., .03)
        for x in self._mel_data:
            plot_line(x)
            cr.stroke()

def main():
    if len(sys.argv) != 3:
        sys.stderr.write('USAGE: vis-mean.py [input mfcc file] [label]\n')
        sys.exit(1)


    in_file = open(sys.argv[1], 'rb') if sys.argv[1] != '-' else sys.stdin
    reader = MFCCReader(in_file)

    label = sys.argv[2]

    profiles, _, frames = reader.read_all()
    profile = profiles[0]

    mel_data = [ zip(profile.mel_freqs[1:], f.mel_powers)
                for f in frames if f.group_header.label == label ]

    vis = PoorPlotter()
    vis.set_data(mel_data = mel_data, label = "mean for label " + label)

    window = gtk.Window()
    window.connect("delete-event", gtk.main_quit)
    window.add(vis)
    window.set_position(gtk.WIN_POS_CENTER)
    window.show_all()

    gtk.main()

if __name__ == '__main__':
    main()
