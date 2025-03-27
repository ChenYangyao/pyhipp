from __future__ import annotations
from ..core.abc import HasSimpleRepr
import numpy as np
from typing import Any, Tuple, Mapping, Union, Iterator, Self, Literal, Optional
from .abc import mpl_axes, MplObj, Artist, mpl_axis
from .artist_formatter import MarkerFormatter, \
    LineFormatter, ErrorBarFormatter, FillFormatter, TextFormatter, FrameFormatter
from .color import Color
from .two_dim import Scatter2D


class PlotFmts:

    def __init__(self) -> None:

        self.line = LineFormatter()
        self.marker = MarkerFormatter(style=None)
        self.errorbar = ErrorBarFormatter()
        self.fill = FillFormatter()

    @property
    def each(self):
        return self.line, self.marker, self.errorbar, self.fill


class Axes(MplObj[mpl_axes.Axes]):

    Raw = mpl_axes.Axes
    ColorSpec = Color.ColorSpec

    def __init__(self, raw: Raw = None, **kw) -> None:
        super().__init__(raw=raw, **kw)

        self._plot_fmts = PlotFmts()
        self._text_fmt = TextFormatter()
        self._frame_fmt = FrameFormatter()

        self._last_draw = []

    def secondary_xaxis(self, location: float | str,
                        functions=None, **mpl_kw) -> Axes:
        out = self._raw.secondary_xaxis(location, functions=functions, **mpl_kw)
        return Axes(out)

    def secondary_yaxis(self, location: float | str,
                        functions=None, **mpl_kw) -> Axes:
        out = self._raw.secondary_yaxis(location, functions=functions, **mpl_kw)
        return Axes(out)

    @property
    def last_draw(self) -> Artist:
        return Artist(self._last_draw[-1])

    def axis_on(self) -> Axes:
        self._raw.set_axis_on()
        return self

    def axis_off(self) -> Axes:
        self._raw.set_axis_off()
        return self

    def grid(self, visible=True, which='major', axis='both',
             lw=None, ls=None, c=None, **mpl_grid_kw):

        kw = dict(which=which, axis=axis, visible=visible,
                  lw=lw, ls=ls, c=c, **mpl_grid_kw
                  )
        self._raw.grid(**kw)

        return self

    def c(self, c: ColorSpec = None, a: float = None) -> Axes:
        for fmt in self._plot_fmts.each:
            fmt.c(c, a)

        return self

    def lim(self, x: Tuple[float, float] = None,
            y: Tuple[float, float] = None) -> Axes:

        return self.fmt_frame(lim={'x': x, 'y': y})

    def label(self, x: str = None, y: str = None) -> Axes:
        '''
        @x:
        @y:
            are wrapped by TextFormatter.
        '''
        fmt = self._text_fmt
        if x is not None:
            x = fmt.wrap_text(x)
        if y is not None:
            y = fmt.wrap_text(y)
        return self.fmt_frame(label={'x': x, 'y': y})

    def scale(self, x: str = None, y: str = None) -> Axes:
        return self.fmt_frame(scale={'x': x, 'y': y})

    def tick_params(self, x: dict = None, y: dict = None,
                    both: dict = None, **kw) -> Axes:
        '''
        Examples
        --------
        ax.tick_params(x={'which': 'both', 'top': False})
        '''
        return self.fmt_frame(tick_params={
            'x': x, 'y': y, 'both': both, **kw})

    def label_outer(self) -> Axes:
        self.fmt_frame(label_outer=True)
        return self

    def fmt(self, frame=None, marker=None, fill=None, line=None,
            errorbar=None, text=None) -> Axes:

        if frame is not None:
            self.fmt_frame(**frame)
        if marker is not None:
            self.fmt_marker(**marker)
        if fill is not None:
            self.fmt_fill(**fill)
        if line is not None:
            self.fmt_line(**line)
        if errorbar is not None:
            self.fmt_errorbar(**errorbar)
        if text is not None:
            self.fmt_text(**text)

        return self

    def fmt_frame(self, **kw) -> Axes:
        fmt = self._frame_fmt
        fmt.update(kw)
        fmt.apply(self)
        return self

    def fmt_marker(self, style=None, c=None, a=None, **kw) -> Axes:
        fmt = self._plot_fmts.marker

        fmt.c(c, a)
        fmt |= {'style': style}

        fmt.update(kw)

        return self

    def fmt_fill(self, c=None, a=None, **kw) -> Axes:

        fmt = self._plot_fmts.fill

        fmt.c(c, a)

        fmt.update(kw)

        return self

    def fmt_line(self, **kw) -> Axes:

        self._plot_fmts.line.update(kw)

        return self

    def fmt_errorbar(self, **kw) -> Axes:

        self._plot_fmts.errorbar.update(kw)

        return self

    def fmt_text(self, **kw) -> Axes:

        self._text_fmt.update(kw)

        return self

    def plot(self, x, y, label=None,
             **mpl_plot_kw) -> Axes:
        '''
        Keywords from properties
        ------------------------
        line: lw, ls.
        marker: marker, ms, mew, mec, mfc.
        '''
        fmt = self._plot_fmts.line
        c = fmt.get_c().get_rgba()
        lw, ls = fmt['lw', 'ls']

        fmt = self._plot_fmts.marker
        marker, ms, mew = fmt['style', 's', 'elw']
        mec, mfc = fmt.get_ec().get_rgba(), fmt.get_fc().get_rgba()

        kw = dict(c=c, lw=lw, ls=ls,
                  marker=marker, ms=ms, mew=mew, mec=mec, mfc=mfc,
                  label=label
                  )
        kw |= mpl_plot_kw

        self._last_draw.append(self._raw.plot(x, y, **kw))

        return self

    def scatter(self, x, y, label=None,
                **mpl_scatter_kw) -> Axes:
        '''
        Keywords from properties
        ------------------------
        marker: marker, s, linewidths, edgecolors, facecolors.
        '''
        fmt = self._plot_fmts.marker

        marker, s, linewidths = fmt['style', 's', 'elw']
        edgecolors, facecolors = fmt.get_ec().get_rgba(), \
            fmt.get_fc().get_rgba()

        kw = dict(
            s=s, linewidths=linewidths,
            marker=marker, edgecolors=edgecolors, facecolors=facecolors,
            label=label
        )
        kw |= mpl_scatter_kw

        self._last_draw.append(self._raw.scatter(x, y, **kw))

        return self

    def scatter_2d(self, x, y, range=None, n_bins=10,
                   weight: np.ndarray = None,  **init_kw):
        '''
        Return a Scatter2D instance for advanced plotting, e.g., mesh, contour, 
        and scatter.
        '''
        return Scatter2D(self, x, y, range=range, n_bins=n_bins,
                         weight=weight, **init_kw)

    def errorbar(self, x, y, yerr=None, xerr=None, label=None,
                 lolims=False, uplims=False, xlolims=False, xuplims=False,
                 **mpl_errorbar_kw) -> Axes:
        '''
        @xerr, yerr: float | array-like (N, ) or (2, N) | None.
            Shape (2, N) for lower (- from y) and upper (+ to y).
            All errors should have positive values.
        
        Keywords from properties
        ------------------------
        line: c, lw, ls.
        errorbar: ecolor, elinewidth, capsize, capthick.
        marker: marker, ms, mew, mec, mfc.
        '''
        fmt = self._plot_fmts.line
        c = fmt.get_c().get_rgba()
        lw, ls = fmt['lw', 'ls']

        fmt = self._plot_fmts.errorbar
        ecolor = fmt.get_c().get_rgba()
        elinewidth, capsize, capthick = fmt['lw', 'capsize', 'capthick']

        fmt = self._plot_fmts.marker
        marker, ms, mew = fmt['style', 's', 'elw']
        mec, mfc = fmt.get_ec().get_rgba(), fmt.get_fc().get_rgba()

        kw = dict(c=c, lw=lw, ls=ls, ecolor=ecolor,
                  elinewidth=elinewidth,
                  capsize=capsize, capthick=capthick,
                  marker=marker, ms=ms, mew=mew, mec=mec, mfc=mfc,
                  label=label, lolims=lolims, uplims=uplims,
                  xlolims=xlolims, xuplims=xuplims)
        kw |= mpl_errorbar_kw

        self._last_draw.append(
            self._raw.errorbar(x, y, yerr=yerr, xerr=xerr, **kw))

        return self

    def fill_between(self, x, y1, y2=0, where=None, interpolate=False,
                     label=None, step=None, **mpl_fill_between_kw):
        '''
        Keywords from properties
        ------------------------
        fill: ec, fc, lw, ls.
        '''
        fmt = self._plot_fmts.fill
        ec, fc = fmt.get_ec().get_rgba(), fmt.get_fc().get_rgba()
        lw, ls = fmt['lw', 'ls']

        kw = dict(
            ec=ec, fc=fc, lw=lw, ls=ls,
            where=where, interpolate=interpolate, step=step,
            label=label,)
        kw |= mpl_fill_between_kw
        self._last_draw.append(self._raw.fill_between(x, y1, y2=y2, **kw))

        return self

    def errorfill(self, x, y, yerr=None, label=None, fill_label=None,
                  fill_between_kw={}, **plot_kw):
        '''
        @yerr: float | array-like (N, ) or (2, N) | None.
            Shape (2, N) for lower (- from y) and upper (+ to y).
            All errors should have positive values.
        @label:
        @fill_label:
            passed to plot() and fill_between(), respectively.
        '''
        if yerr is None:
            ylo = yhi = None
        elif np.isscalar(yerr):
            ylo = yhi = yerr
        else:
            yerr = np.asarray(yerr)
            if yerr.ndim == 1:
                ylo = yhi = yerr
            else:
                ylo, yhi = yerr
        y = np.asarray(y)
        ylo, yhi = y - ylo, y + yhi

        self.plot(x, y, label=label, **plot_kw)

        kw = {'label': fill_label}
        kw |= fill_between_kw
        self.fill_between(x, ylo, yhi, **kw)

        return self

    def stairs(self, values, edges=None, orientation='vertical',
               baseline=0, fill=True, label=None, **mpl_stairs_kw):
        '''
        Keywords from properties
        ------------------------
        fill: ec, fc, lw, ls.
        '''
        fmt = self._plot_fmts.fill
        ec, fc = fmt.get_ec().get_rgba(), fmt.get_fc().get_rgba()
        lw, ls = fmt['lw', 'ls']

        kw = dict(orientation=orientation,
                  baseline=baseline, fill=fill, label=label,
                  ec=ec, fc=fc, lw=lw, ls=ls,)
        kw |= mpl_stairs_kw

        self._last_draw.append(
            self._raw.stairs(values, edges, **kw)
        )

        return self

    def hist(self, x, bins=None, range=None, density=False,
             weights=None, histtype='stepfilled',
             log=False, align='mid', rwidth=None,
             orientation='vertical',
             **mpl_hist_kw):
        '''
        Keywords from properties
        ------------------------
        fill: ec, fc, lw, ls.
        '''
        fmt = self._plot_fmts.fill
        ec, fc = fmt.get_ec().get_rgba(), fmt.get_fc().get_rgba()
        lw, ls = fmt['lw', 'ls']

        kw = dict(
            bins=bins, range=range, density=density, weights=weights,
            ec=ec, fc=fc, lw=lw, ls=ls,
            histtype=histtype, log=log, align=align, rwidth=rwidth,
            orientation=orientation,)
        kw |= mpl_hist_kw

        self._last_draw.append(
            self._raw.hist(x, **kw)
        )

        return self

    def text(self, content, coords, va='center', ha='left',
             **mpl_text_kw) -> Axes:
        '''
        Keywords from properties
        ------------------------
        text: color, fontsize.
        '''
        fmt = self._text_fmt
        color = fmt.get_c().get_rgba()
        fontsize = fmt['s']

        kw = dict(
            transform=self._raw.transAxes,
            color=color, fontsize=fontsize, va=va, ha=ha,)
        kw |= mpl_text_kw
        content = fmt.wrap_text(content)

        self._last_draw.append(self._raw.text(*coords, content, **kw))

        return self

    def leg(self, *args, loc='best', fontsize=None,
            markerscale=None, numpoints=None, scatterpoints=None,
            handlelength=None, handleheight=None, handletextpad=None,
            borderpad=None, borderaxespad=None,
            ncol=1, labelspacing=None, columnspacing=None,
            title=None, title_fontsize=None,
            frameon=None, framealpha=None,
            labelcolor: Literal['linecolor', 'mec', 'mfc'] | None = None,
            **mpl_legend_kw):

        loc_map = {
            'r': 'right',
            'lr': 'lower right', 'll': 'lower left', 'cl': 'center left',
            'ur': 'upper right', 'ul': 'upper left', 'uc': 'upper center',
            'center left': 'cl', 'center right': 'cr', 'c': 'center',
        }
        loc = loc_map.get(loc, loc)

        kw = dict(loc=loc,
                  fontsize=fontsize,
                  markerscale=markerscale,
                  numpoints=numpoints,
                  scatterpoints=scatterpoints,
                  handlelength=handlelength,
                  handleheight=handleheight, handletextpad=handletextpad,
                  title=title,
                  title_fontsize=title_fontsize,
                  borderpad=borderpad, borderaxespad=borderaxespad,
                  ncol=ncol,
                  labelspacing=labelspacing,
                  columnspacing=columnspacing,
                  frameon=frameon,
                  framealpha=framealpha,
                  labelcolor=labelcolor) | mpl_legend_kw

        lt = self._raw.legend(*args, **kw)
        self._last_draw.append(lt)

        return self


class AxesArray(HasSimpleRepr):

    Raw = 'np.ndarray[Any, Axes.Raw]'

    def __init__(self, axes_array: Raw = None, **kw) -> None:

        super().__init__(**kw)

        if isinstance(axes_array, AxesArray):
            arr = axes_array._axes_array
        else:
            arr = np.asarray(axes_array)
        shape = arr.shape

        arr = [(ax if isinstance(ax, Axes) else Axes(ax))
               for ax in arr.flat]
        arr = np.array(arr).reshape(shape)

        self._axes_array = arr

    def to_simple_repr(self) -> list:
        return self._axes_array.tolist()

    def to_list(self, flat=True) -> list[Axes]:
        axs = self._axes_array
        if flat:
            axs = axs.ravel()
        return axs.tolist()

    def __getitem__(self, key) -> Union[Axes, AxesArray]:
        val = self._axes_array[key]
        if not isinstance(val, Axes):
            val = AxesArray(val)
        return val

    def __len__(self):
        return len(self._axes_array)

    def __iter__(self) -> Union[Iterator[Axes], Iterator[AxesArray]]:
        n = len(self)
        for i in range(n):
            yield self[i]

    @property
    def flat(self):
        return AxesArray([ax for ax in self._axes_array.flat])

    @property
    def T(self):
        return AxesArray(self._axes_array.T)

    @property
    def size(self):
        return self._axes_array.size

    @property
    def shape(self):
        return self._axes_array.shape

    def grid(self, visible=True, which='major', axis='both',
             lw=None, ls=None, c=None, **mpl_grid_kw):

        kw = {
            'visible': visible, 'which': which, 'axis': axis,
            'lw': lw, 'ls': ls, 'c': c, **mpl_grid_kw
        }
        for ax in self:
            ax.grid(**kw)

        return self

    def c(self, c: Color.ColorSpec = None, a: float = None) -> Self:
        for ax in self:
            ax.c(c=c, a=a)
        return self

    def lim(self, x: Tuple[float, float] = None,
            y: Tuple[float, float] = None) -> AxesArray:
        for ax in self:
            ax.lim(x, y)
        return self

    def label(self, x: str = None, y: str = None) -> AxesArray:
        for ax in self:
            ax.label(x, y)
        return self

    def scale(self, x: str = None, y: str = None) -> AxesArray:
        for ax in self:
            ax.scale(x, y)
        return self

    def tick_params(self, x: dict = None, y: dict = None,
                    both: dict = None, **kw) -> Self:
        for ax in self:
            ax.tick_params(x, y, both, **kw)
        return self

    def label_outer(self):
        for ax in self:
            ax.label_outer()
        return self

    def fmt(self, frame=None, marker=None, fill=None, line=None,
            errorbar=None, text=None) -> AxesArray:

        kw = {
            'frame': frame, 'marker': marker, 'fill': fill,
            'line': line, 'errorbar': errorbar, 'text': text,
        }
        for ax in self:
            ax.fmt(**kw)

        return self

    def fmt_frame(self, **kw) -> AxesArray:
        for ax in self:
            ax.fmt_frame(**kw)
        return self

    def fmt_marker(self, style=None, c=None, a=None, **kw) -> AxesArray:
        kw |= {
            'style': style, 'c': c, 'a': a,
        }
        for ax in self:
            ax.fmt_marker(**kw)
        return self

    def fmt_fill(self, c=None, a=None, **kw) -> AxesArray:
        kw |= {'c': c, 'a': a}
        for ax in self:
            ax.fmt_fill(**kw)
        return self

    def fmt_line(self, **kw) -> AxesArray:
        for ax in self:
            ax.fmt_line(**kw)
        return self

    def fmt_errorbar(self, **kw) -> AxesArray:
        for ax in self:
            ax.fmt_errorbar(**kw)
        return self

    def fmt_text(self, **kw) -> AxesArray:
        for ax in self:
            ax.fmt_text(**kw)
        return self

    def leg(self, *args, **kw) -> AxesArray:
        for ax in self:
            ax.leg(*args, **kw)

        return self
