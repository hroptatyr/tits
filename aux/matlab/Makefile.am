### Makefile.am
include $(top_builddir)/version.mk

AM_CFLAGS = $(EXTRA_CFLAGS)
AM_CPPFLAGS = -D_POSIX_C_SOURCE=200112L -D_XOPEN_SOURCE=600 -D_BSD_SOURCE
AM_LDFLAGS = $(XCCLDFLAGS)
EXTRA_DIST =

if BUILD_MATCLI
mexdir = $(MATLABPATH)/tits
octlab_CFLAGS = -DMATLAB_MEX_FILE $(matlab_CFLAGS)
else  !BUILD_MATCLI
## no idea what we're doing here
mexdir = $(pkglibexecdir)
endif  !BUILD_MATCLI

EXTRA_DIST += $(mex_DATA)
mex_LTLIBRARIES =
mex_LDFLAGS = -module -shared -avoid-version -shrext ".mex"
mex_DATA =

mex_LTLIBRARIES += tits_xcor.la
mex_DATA += tits_xcor.m
tits_xcor_la_SOURCES = tits_xcor.c
tits_xcor_la_SOURCES += mex.h
tits_xcor_la_CPPFLAGS = $(AM_CPPFLAGS)
tits_xcor_la_CPPFLAGS += $(octlab_CFLAGS)
tits_xcor_la_CPPFLAGS += -I$(top_srcdir)/src
tits_xcor_la_LDFLAGS = $(AM_LDFLAGS)
tits_xcor_la_LDFLAGS += $(mex_LDFLAGS)
tits_xcor_la_LDFLAGS += $(XCCLDFLAGS)
tits_xcor_la_LIBADD = $(top_builddir)/src/libtits.la


## matlab needs mexa64, octave needs mex
install-data-hook:
	-for i in ute_tick ute_nticks ute_slut ute_open ute_close; do \
		cd $(DESTDIR)$(mexdir) && mv -f $${i}.mex $${i}.$(MEXEXT); \
	done

uninstall-hook:
	-for i in ute_tick ute_nticks ute_slut ute_open ute_close; do \
		$(RM) -- $(DESTDIR)$(mexdir)/$${i}.$(MEXEXT); \
	done


## Create preprocessor output (debugging purposes only)
.c.i:
	$(COMPILE) -E -o $@ $<

## Create assembler output (debugging purposes only)
.c.s:
	$(COMPILE) -S -c $(AM_CFLAGS) $<
