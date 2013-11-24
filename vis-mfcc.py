#!/usr/bin/python

import sys, math
import gtk, glib
from common import MFCCReader

class PoorPlotter(gtk.DrawingArea):

    def __init__(self):
        super(PoorPlotter, self).__init__()
        self.connect("expose_event", self.expose)
        self.set_size_request(1024,256)

        self._margins = (10.5,15.5,20.5,30.5)
        self._xtics = tuple(xrange(0,8001,1000))
        self._ytics = tuple(xrange(-10,-141,-10))
        self._xrange = (0,8000)
        self._yrange = (-10,-140)
        self._ticksize = 5
        self._labeloffs = (5,5)
        self._dotradius = 3

        self._width = self._height = 1

        self._mel_data = ()
        self._freq_data = ()
        self._frameno = None

    def set_data(self, mel_data=None, freq_data=None, frameno=None):
        if mel_data is not None:
            self._mel_data = mel_data
        if freq_data is not None:
            self._freq_data = freq_data
        if frameno is not None:
            self._frameno = frameno

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

        if self._frameno is not None:
            label = 'Frame %d' % self._frameno
            extents = cr.text_extents(label)

            px,py = self.pt2xy((self._xrange[1], self._yrange[0]))
            cr.move_to(px - self._labeloffs[0] - extents[2], py + self._labeloffs[1] + extents[3])
            cr.show_text(label)

        def plot_line(data):
            for idx,(x,y) in enumerate(data):
                px,py = self.pt2xy((x,y))
                if 0 == idx:
                    cr.move_to(px,py)
                else:
                    cr.line_to(px,py)

        cr.set_source_rgb(0.5, 0.5, 0.5)
        plot_line(self._freq_data)
        cr.stroke()

        cr.set_source_rgb(0., 0., 0.)
        cr.set_line_width(2.)
        plot_line(self._mel_data)
        cr.stroke()

        for x,y in self._mel_data:
            px,py = self.pt2xy((x,y))
            cr.new_sub_path()
            cr.arc(px,py, self._dotradius, 0, 2.*math.pi)
        cr.fill()

class MFCCWatcher(object):
    def __init__(self, vis):
        self._vis = vis
        self._reader = None
        self._frameno = 0
        self.push_data = True

    def data_available(self, source, condition):
        if self._reader is None:
            self._reader = MFCCReader(source)
            return True

        return self.forward()

    def forward(self):
        try:
            mel, freq = next(self._reader)
        except StopIteration:
            return False

        self._frameno += 1
        mel = zip(self._reader.mel_freqs[1:], mel)
        freq = zip(self._reader.fft_freqs, freq)

        if self.push_data:
            self._vis.set_data(mel_data = mel, freq_data = freq, frameno = self._frameno)
            self._vis.queue_draw()
        return True

    def backward(self):
        if self._frameno <= 1:
            return False
        self._reader.seek(-2)
        self._frameno -= 2
        return self.forward()

def main():
    vis = PoorPlotter()

    try:
        sys.stdin.seek(0, 1)
        stdin_seekable = True
    except IOError:
        stdin_seekable = False;

    watcher = MFCCWatcher(vis)
    if not stdin_seekable:
        glib.io_add_watch(sys.stdin, glib.IO_IN, watcher.data_available)
    else:
        watcher.data_available(sys.stdin, None)

    def keypress(widget, event):
        if not stdin_seekable:
            if event.keyval == gtk.keysyms.space:
                watcher.push_data = not watcher.push_data
        else:
            if event.keyval == gtk.keysyms.Left:
                watcher.backward()
            elif event.keyval == gtk.keysyms.Right:
                watcher.forward()

    window = gtk.Window()
    window.connect("delete-event", gtk.main_quit)
    window.connect("key-press-event", keypress)
    window.add(vis)
    window.set_position(gtk.WIN_POS_CENTER)
    window.show_all()

    gtk.main()

if __name__ == '__main__':
    main()
