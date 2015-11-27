### Makefile.am
include $(top_builddir)/version.mk

AM_CFLAGS = $(EXTRA_CFLAGS)
AM_CPPFLAGS = -D_POSIX_C_SOURCE=201001L -D_XOPEN_SOURCE=700 -D_BSD_SOURCE
AM_LDFLAGS = $(XCCLDFLAGS)

bin_PROGRAMS =
noinst_PROGRAMS =
lib_LTLIBRARIES =
noinst_LTLIBRARIES =
pkglib_LTLIBRARIES =
noinst_HEADERS =
BUILT_SOURCES =
EXTRA_DIST = $(BUILT_SOURCES)
CLEANFILES = 
SUFFIXES =

CLEANFILES += version.c
EXTRA_DIST += version.c.in

if BUILD_MATCLI
mexdir = $(MATLABPATH)/cotse
octlab_CFLAGS = -DMATLAB_MEX_FILE $(matlab_CFLAGS)
else  !BUILD_MATCLI
## no idea what we're doing here
mexdir = $(pkglibexecdir)
endif  !BUILD_MATCLI

EXTRA_DIST += $(mex_DATA)
mex_LTLIBRARIES =
mex_LDFLAGS = -module -shared -avoid-version -shrext ".mexa64"
mex_DATA =

mkl_CFLAGS = -I/opt/intel/mkl/include -mkl -xHost
mkl_LIBS =

noinst_LTLIBRARIES += libcotse.la
libcotse_la_SOURCES = cotse.h
libcotse_la_SOURCES += hash.c hash.h
libcotse_la_SOURCES += version.c version.h
EXTRA_libcotse_la_DEPENDENCIES = $(LTDLDEPS)

EXTRA_DIST += crosscorrirr.m

if BUILD_MATCLI
mex_LTLIBRARIES += crosscorrirr_.la
mex_DATA += crosscorrirr_.m
crosscorrirr__la_SOURCES = crosscorrirr_.c
crosscorrirr__la_CPPFLAGS = $(AM_CPPFLAGS)
crosscorrirr__la_CPPFLAGS += $(octlab_CFLAGS)
crosscorrirr__la_CPPFLAGS += $(mkl_CFLAGS)
crosscorrirr__la_LDFLAGS = $(AM_LDFLAGS)
crosscorrirr__la_LDFLAGS += $(mex_LDFLAGS)
crosscorrirr__la_LDFLAGS += $(mkl_LIBS)
crosscorrirr__la_LDFLAGS += $(XCCLDFLAGS)
endif  BUILD_MATCLI

noinst_PROGRAMS += guess_lag
guess_lag_SOURCES = guess_lag.c guess_lag.yuck
guess_lag_CPPFLAGS = $(AM_CPPFLAGS)
guess_lag_CPPFLAGS += -DSTANDALONE
guess_lag_CPPFLAGS += $(mkl_CFLAGS)
guess_lag_CPPFLAGS += $(dfp754_CFLAGS)
guess_lag_LDFLAGS = $(AM_LDFLAGS)
guess_lag_LDFLAGS += $(mkl_LIBS)
guess_lag_LDFLAGS += $(dfp754_LIBS)
guess_lag_LDADD = libcotse.la
BUILT_SOURCES += guess_lag.yucc


## version rules
version.c: version.c.in $(top_builddir)/.version
	$(AM_V_GEN) PATH="$(top_builddir)/build-aux:$${PATH}" \
		yuck$(EXEEXT) scmver --ignore-noscm --force -o $@ \
			--use-reference --reference $(top_builddir)/.version $<

## yuck rule
SUFFIXES += .yuck
SUFFIXES += .yucc
.yuck.yucc:
	$(AM_V_GEN) PATH=$(top_builddir)/build-aux:"$${PATH}" \
		yuck$(EXEEXT) gen -o $@ $<

## Create preprocessor output (debugging purposes only)
.c.i:
	$(COMPILE) -E -o $@ $<

## Create assembler output (debugging purposes only)
.c.s:
	$(COMPILE) -S -c $(AM_CFLAGS) $<
