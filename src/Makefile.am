### Makefile.am
include $(top_builddir)/version.mk

AM_CFLAGS = $(EXTRA_CFLAGS)
AM_CPPFLAGS = -D_POSIX_C_SOURCE=201001L -D_XOPEN_SOURCE=700 -D_BSD_SOURCE
AM_CPPFLAGS += -D_HAVE_VERSION_H
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

MKL_CFLAGS = -I/opt/intel/mkl/include -mkl -xHost
MKL_LIBS = -L/opt/intel/mkl/lib/intel64 -Wl,-rpath,/opt/intel/mkl/lib/intel64 -lmkl_rt

noinst_LTLIBRARIES += libcotse.la
libcotse_la_SOURCES = cotse.h
libcotse_la_SOURCES += hash.c hash.h
libcotse_la_SOURCES += version.c version.h
EXTRA_libcotse_la_DEPENDENCIES = $(LTDLDEPS)

noinst_PROGRAMS += crosscorrirr
crosscorrirr_SOURCES = crosscorrirr.c crosscorrirr.yuck
crosscorrirr_CPPFLAGS = $(AM_CPPFLAGS)
crosscorrirr_CPPFLAGS += -DSTANDALONE
crosscorrirr_CPPFLAGS += $(MKL_CFLAGS)
crosscorrirr_LDFLAGS = $(AM_LDFLAGS)
crosscorrirr_LDFLAGS += $(MKL_LIBS)
BUILT_SOURCES += crosscorrirr.yucc


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
