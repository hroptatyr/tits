# Help the Developers and yourself. Just use the C locale and settings
# for the compilation. They can still be overriden by make LANG=<whatever>
# but that is general a not very good idea
include $(top_builddir)/version.mk

LANG = C
LC_ALL = C

AM_CFLAGS = $(EXTRA_CFLAGS)
AM_CPPFLAGS = -D_POSIX_C_SOURCE=200112L -D_XOPEN_SOURCE=600 -D_BSD_SOURCE
AM_CPPFLAGS += -DTEST

EXTRA_DIST = $(BUILT_SOURCES) $(TESTS)
TESTS =
TEST_EXTENSIONS =
BUILT_SOURCES =
bin_tests =
batch_tests =
check_PROGRAMS =
CLEANFILES = $(check_PROGRAMS)

tits_CFLAGS = -I$(top_srcdir)/src
tits_LIBS = $(top_builddir)/src/libtits.la

gsl_CFLAGS := $(shell gsl-config --cflags)
gsl_LIBS := $(shell gsl-config --libs)

check_PROGRAMS += $(bin_tests)
TESTS += $(bin_tests)

bin_tests += sacf_01
sacf_01_CPPFLAGS = $(AM_CPPFLAGS)
sacf_01_CPPFLAGS += $(gsl_CFLAGS)
sacf_01_CPPFLAGS += $(tits_CFLAGS)
sacf_01_LDFLAGS = $(AM_LDFLAGS)
sacf_01_LDFLAGS += $(gsl_LIBS)
sacf_01_LDFLAGS += $(tits_LIBS)

bin_tests += sacf_02
sacf_02_CPPFLAGS = $(AM_CPPFLAGS)
sacf_02_CPPFLAGS += $(gsl_CFLAGS)
sacf_02_CPPFLAGS += $(tits_CFLAGS)
sacf_02_LDFLAGS = $(AM_LDFLAGS)
sacf_02_LDFLAGS += $(gsl_LIBS)
sacf_02_LDFLAGS += $(tits_LIBS)

bin_tests += sacf_03
sacf_03_CPPFLAGS = $(AM_CPPFLAGS)
sacf_03_CPPFLAGS += $(gsl_CFLAGS)
sacf_03_CPPFLAGS += $(tits_CFLAGS)
sacf_03_LDFLAGS = $(AM_LDFLAGS)
sacf_03_LDFLAGS += $(gsl_LIBS)
sacf_03_LDFLAGS += $(tits_LIBS)

bin_tests += sacf_04
sacf_04_CPPFLAGS = $(AM_CPPFLAGS)
sacf_04_CPPFLAGS += $(gsl_CFLAGS)
sacf_04_CPPFLAGS += $(tits_CFLAGS)
sacf_04_LDFLAGS = $(AM_LDFLAGS)
sacf_04_LDFLAGS += $(gsl_LIBS)
sacf_04_LDFLAGS += $(tits_LIBS)

bin_tests += acf24ar_01
acf24ar_01_CPPFLAGS = $(AM_CPPFLAGS)
acf24ar_01_CPPFLAGS += $(tits_CFLAGS)
acf24ar_01_LDFLAGS = $(AM_LDFLAGS)
acf24ar_01_LDFLAGS += $(tits_LIBS)

bin_tests += acf24ar_02
acf24ar_02_CPPFLAGS = $(AM_CPPFLAGS)
acf24ar_02_CPPFLAGS += $(tits_CFLAGS)
acf24ar_02_LDFLAGS = $(AM_LDFLAGS)
acf24ar_02_LDFLAGS += $(tits_LIBS)

bin_tests += acf24ar_03
acf24ar_03_CPPFLAGS = $(AM_CPPFLAGS)
acf24ar_03_CPPFLAGS += $(tits_CFLAGS)
acf24ar_03_LDFLAGS = $(AM_LDFLAGS)
acf24ar_03_LDFLAGS += $(tits_LIBS)

bin_tests += acf24ar_04
acf24ar_04_CPPFLAGS = $(AM_CPPFLAGS)
acf24ar_04_CPPFLAGS += $(tits_CFLAGS)
acf24ar_04_LDFLAGS = $(AM_LDFLAGS)
acf24ar_04_LDFLAGS += $(tits_LIBS)

bin_tests += roots_01
roots_01_CPPFLAGS = $(AM_CPPFLAGS)
roots_01_CPPFLAGS += $(tits_CFLAGS)
roots_01_LDFLAGS = $(AM_LDFLAGS)
roots_01_LDFLAGS += $(tits_LIBS)

bin_tests += roots_02
roots_02_CPPFLAGS = $(AM_CPPFLAGS)
roots_02_CPPFLAGS += $(tits_CFLAGS)
roots_02_LDFLAGS = $(AM_LDFLAGS)
roots_02_LDFLAGS += $(tits_LIBS)

bin_tests += roots_03
roots_03_CPPFLAGS = $(AM_CPPFLAGS)
roots_03_CPPFLAGS += $(tits_CFLAGS)
roots_03_LDFLAGS = $(AM_LDFLAGS)
roots_03_LDFLAGS += $(tits_LIBS)

bin_tests += roots_04
roots_04_CPPFLAGS = $(AM_CPPFLAGS)
roots_04_CPPFLAGS += $(tits_CFLAGS)
roots_04_LDFLAGS = $(AM_LDFLAGS)
roots_04_LDFLAGS += $(tits_LIBS)

## Makefile.am ends here
