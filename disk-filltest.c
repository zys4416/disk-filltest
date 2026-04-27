/******************************************************************************
 * disk-filltest.c - Program to fill a hard disk with random data
 *
 * Usage: ./disk-filltest
 *
 * The program will fill the current directory with files called random-#####.
 * Each file is up to 1 GiB in size and contains randomly generated integers.
 * When the disk is full, writing is finished and all files are read from disk.
 * During reading the file contents is checked against the pseudo-random
 * sequence to detect changed data blocks. Any unexpected value will output
 * an error. Reading and writing speed are shown during operation.
 *
 ******************************************************************************
 * Copyright (C) 2012-2020 Timo Bingmann <tb@panthema.net>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#define VERSION "0.8.3b"

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#if defined(_MSC_VER) || defined(__MINGW32__)
  /* no <sys/statvfs.h> */
#else
  #include <sys/statvfs.h>
  #define HAVE_STATVFS 1
#endif

/* Detect x86/x86_64 for AVX2 support */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #define HAVE_X86 1
#else
  #define HAVE_X86 0
#endif

#if HAVE_X86
  #include <immintrin.h>
#endif

/* random seed used */
unsigned int g_seed;

/* only perform read operation */
int gopt_readonly = 0;

/* immediately unlink files after open */
int gopt_unlink_immediate = 0;

/* unlink files after complete run */
int gopt_unlink_after = 0;

/* skip file verification (e.g. for wiping a disk) */
int gopt_skip_verify = 0;

/* individual file size in MiB */
unsigned int gopt_file_size = 0;

/* file number limit */
unsigned int gopt_file_limit = UINT_MAX;

/* number of repetitions */
int gopt_repeat = 1;

/* delay in seconds after each file (except the last) */
int gopt_delay = 0;

/* delay in seconds between -R repetitions (except after the last) */
int gopt_repeat_delay = 0;

/* size of last file written */
unsigned int g_last_filesize = UINT_MAX;

/* return the current timestamp */
double timestamp(void)
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return ((double)(tv.tv_sec) + (double)(tv.tv_usec/1e6));
}

/* simple linear congruential random generator, faster than rand() and totally
 * sufficient for this cause. */
uint64_t lcg_random(uint64_t *xn)
{
    *xn = 0x27BB2EE687B0B0FDLLU * *xn + 0xB504F32DLU;
    return *xn;
}

/* Multi-lane LCG constants for 4-lane parallel generation.
 * a_4 = a^4 mod 2^64, c_4 = c*(1 + a + a^2 + a^3) mod 2^64
 * where a = 0x27BB2EE687B0B0FD, c = 0xB504F32D.
 * Each lane advances 4 steps at a time: x_{n+4} = a_4 * x_n + c_4 */
#define LCG_A  0x27BB2EE687B0B0FDLLU
#define LCG_C  0xB504F32DLU
#define LCG_A4 0x8DD29D8705EB5451LLU
#define LCG_C4 0x7BCFDF788DC37E7CLLU
#define LCG_LANES 4

/* item type used in blocks written to disk */
typedef uint64_t item_type;

/* Fill a block with LCG pseudo-random data using 4 parallel lanes.
 * Produces byte-identical output to the sequential lcg_random() loop.
 * Returns the final LCG state (equal to block[count-1]). */
uint64_t lcg_fill_block(item_type *block, unsigned int count, uint64_t rnd)
{
    uint64_t lane0, lane1, lane2, lane3;
    unsigned int i;

    /* Prime the 4 lanes by advancing sequentially */
    lane0 = LCG_A * rnd   + LCG_C;
    lane1 = LCG_A * lane0 + LCG_C;
    lane2 = LCG_A * lane1 + LCG_C;
    lane3 = LCG_A * lane2 + LCG_C;

    block[0] = lane0;
    block[1] = lane1;
    block[2] = lane2;
    block[3] = lane3;

    /* Main loop: advance all 4 lanes in parallel */
    for (i = LCG_LANES; i < count; i += LCG_LANES)
    {
        lane0 = LCG_A4 * lane0 + LCG_C4;
        lane1 = LCG_A4 * lane1 + LCG_C4;
        lane2 = LCG_A4 * lane2 + LCG_C4;
        lane3 = LCG_A4 * lane3 + LCG_C4;

        block[i + 0] = lane0;
        block[i + 1] = lane1;
        block[i + 2] = lane2;
        block[i + 3] = lane3;
    }

    return lane3;
}

/* Verify a block against the expected LCG sequence using 4 parallel lanes.
 * Returns 0 on success, or (i+1) where i is the index of the first mismatch.
 * Updates *rnd_out to the final LCG state on success. */
unsigned int lcg_check_block(const item_type *block, unsigned int count,
                             uint64_t rnd, uint64_t *rnd_out)
{
    uint64_t lane0, lane1, lane2, lane3;
    unsigned int i;
    unsigned int count_aligned = count - (count % LCG_LANES);

    /* Very short block: fall back to sequential */
    if (count < LCG_LANES) {
        for (i = 0; i < count; ++i) {
            rnd = LCG_A * rnd + LCG_C;
            if (block[i] != rnd) return i + 1;
        }
        *rnd_out = rnd;
        return 0;
    }

    /* Prime the 4 lanes */
    lane0 = LCG_A * rnd   + LCG_C;
    lane1 = LCG_A * lane0 + LCG_C;
    lane2 = LCG_A * lane1 + LCG_C;
    lane3 = LCG_A * lane2 + LCG_C;

    if (block[0] != lane0) return 1;
    if (block[1] != lane1) return 2;
    if (block[2] != lane2) return 3;
    if (block[3] != lane3) return 4;

    /* Main loop */
    for (i = LCG_LANES; i < count_aligned; i += LCG_LANES)
    {
        lane0 = LCG_A4 * lane0 + LCG_C4;
        lane1 = LCG_A4 * lane1 + LCG_C4;
        lane2 = LCG_A4 * lane2 + LCG_C4;
        lane3 = LCG_A4 * lane3 + LCG_C4;

        if (block[i + 0] != lane0) return i + 1;
        if (block[i + 1] != lane1) return i + 2;
        if (block[i + 2] != lane2) return i + 3;
        if (block[i + 3] != lane3) return i + 4;
    }

    /* Tail: handle remaining 1-3 elements sequentially */
    rnd = lane3;
    for (i = count_aligned; i < count; ++i) {
        rnd = LCG_A * rnd + LCG_C;
        if (block[i] != rnd) return i + 1;
    }

    *rnd_out = (count == count_aligned) ? lane3 : rnd;
    return 0;
}

/* Function pointer types for LCG block operations */
typedef uint64_t (*lcg_fill_block_fn)(item_type *block, unsigned int count,
                                      uint64_t rnd);
typedef unsigned int (*lcg_check_block_fn)(const item_type *block,
                                           unsigned int count,
                                           uint64_t rnd, uint64_t *rnd_out);

/* Active function pointers, initialized by lcg_init() */
static lcg_fill_block_fn  lcg_fill  = lcg_fill_block;
static lcg_check_block_fn lcg_check = lcg_check_block;

#if HAVE_X86

__attribute__((target("avx2")))
static uint64_t lcg_fill_block_avx2(item_type *block, unsigned int count,
                                     uint64_t rnd)
{
    __m256i lanes, a4_vec, a4_hi_vec, c4_vec;
    uint64_t lane0, lane1, lane2, lane3;
    unsigned int i;

    /* Prime the 4 lanes sequentially (same as scalar) */
    lane0 = LCG_A * rnd   + LCG_C;
    lane1 = LCG_A * lane0 + LCG_C;
    lane2 = LCG_A * lane1 + LCG_C;
    lane3 = LCG_A * lane2 + LCG_C;

    block[0] = lane0;
    block[1] = lane1;
    block[2] = lane2;
    block[3] = lane3;

    /* Pack lanes into a 256-bit register */
    lanes = _mm256_set_epi64x((long long)lane3, (long long)lane2,
                              (long long)lane1, (long long)lane0);

    /* Broadcast constants, precompute a4_hi outside loop */
    a4_vec    = _mm256_set1_epi64x((long long)LCG_A4);
    a4_hi_vec = _mm256_srli_epi64(a4_vec, 32);
    c4_vec    = _mm256_set1_epi64x((long long)LCG_C4);

    /* Main AVX2 loop: emulate 64x64->64 multiply with 32x32->64 ops.
     * a*x mod 2^64 = a_lo*x_lo + (a_lo*x_hi + a_hi*x_lo) << 32 */
    for (i = LCG_LANES; i < count; i += LCG_LANES)
    {
        __m256i lo_lo, lo_hi, hi_lo, cross, cross_shifted, lanes_hi;

        lo_lo    = _mm256_mul_epu32(a4_vec, lanes);
        lanes_hi = _mm256_srli_epi64(lanes, 32);
        lo_hi    = _mm256_mul_epu32(a4_vec, lanes_hi);
        hi_lo    = _mm256_mul_epu32(a4_hi_vec, lanes);

        cross         = _mm256_add_epi64(lo_hi, hi_lo);
        cross_shifted = _mm256_slli_epi64(cross, 32);
        lanes         = _mm256_add_epi64(lo_lo, cross_shifted);
        lanes         = _mm256_add_epi64(lanes, c4_vec);

        _mm256_storeu_si256((__m256i *)(block + i), lanes);
    }

    return (uint64_t)_mm256_extract_epi64(lanes, 3);
}

__attribute__((target("avx2")))
static unsigned int lcg_check_block_avx2(const item_type *block,
                                          unsigned int count,
                                          uint64_t rnd, uint64_t *rnd_out)
{
    __m256i lanes, a4_vec, a4_hi_vec, c4_vec;
    uint64_t lane0, lane1, lane2, lane3;
    unsigned int i;
    unsigned int count_aligned = count - (count % LCG_LANES);

    /* Very short block: fall back to sequential */
    if (count < LCG_LANES) {
        for (i = 0; i < count; ++i) {
            rnd = LCG_A * rnd + LCG_C;
            if (block[i] != rnd) return i + 1;
        }
        *rnd_out = rnd;
        return 0;
    }

    /* Prime the 4 lanes sequentially */
    lane0 = LCG_A * rnd   + LCG_C;
    lane1 = LCG_A * lane0 + LCG_C;
    lane2 = LCG_A * lane1 + LCG_C;
    lane3 = LCG_A * lane2 + LCG_C;

    if (block[0] != lane0) return 1;
    if (block[1] != lane1) return 2;
    if (block[2] != lane2) return 3;
    if (block[3] != lane3) return 4;

    /* Pack lanes into SIMD register */
    lanes = _mm256_set_epi64x((long long)lane3, (long long)lane2,
                              (long long)lane1, (long long)lane0);

    a4_vec    = _mm256_set1_epi64x((long long)LCG_A4);
    a4_hi_vec = _mm256_srli_epi64(a4_vec, 32);
    c4_vec    = _mm256_set1_epi64x((long long)LCG_C4);

    /* Main AVX2 loop with comparison */
    for (i = LCG_LANES; i < count_aligned; i += LCG_LANES)
    {
        __m256i lo_lo, lo_hi, hi_lo, cross, cross_shifted, lanes_hi;
        __m256i loaded, cmp;
        int mask;

        lo_lo    = _mm256_mul_epu32(a4_vec, lanes);
        lanes_hi = _mm256_srli_epi64(lanes, 32);
        lo_hi    = _mm256_mul_epu32(a4_vec, lanes_hi);
        hi_lo    = _mm256_mul_epu32(a4_hi_vec, lanes);

        cross         = _mm256_add_epi64(lo_hi, hi_lo);
        cross_shifted = _mm256_slli_epi64(cross, 32);
        lanes         = _mm256_add_epi64(lo_lo, cross_shifted);
        lanes         = _mm256_add_epi64(lanes, c4_vec);

        loaded = _mm256_loadu_si256((const __m256i *)(block + i));
        cmp    = _mm256_cmpeq_epi64(lanes, loaded);
        mask   = _mm256_movemask_epi8(cmp);

        if (mask != (int)0xFFFFFFFF) {
            if ((mask & 0xFF) != 0xFF)           return i + 1;
            if ((mask & 0xFF00) != 0xFF00)       return i + 2;
            if ((mask & 0xFF0000) != 0xFF0000)   return i + 3;
            return i + 4;
        }
    }

    /* Tail: handle remaining 1-3 elements sequentially */
    rnd = (uint64_t)_mm256_extract_epi64(lanes, 3);
    for (i = count_aligned; i < count; ++i) {
        rnd = LCG_A * rnd + LCG_C;
        if (block[i] != rnd) return i + 1;
    }

    *rnd_out = (count == count_aligned)
        ? (uint64_t)_mm256_extract_epi64(lanes, 3) : rnd;
    return 0;
}

#endif /* HAVE_X86 */

/* Initialize LCG function pointers based on CPU capabilities */
void lcg_init(void)
{
#if HAVE_X86
  #if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2")) {
        /* Self-test: verify AVX2 produces identical output to scalar */
        item_type test_scalar[256], test_avx2[256];
        uint64_t seed = 0x123456789ABCDEF0ULL;

        lcg_fill_block(test_scalar, 256, seed);
        lcg_fill_block_avx2(test_avx2, 256, seed);

        if (memcmp(test_scalar, test_avx2, sizeof(test_scalar)) != 0) {
            fprintf(stderr, "AVX2 self-test failed, using scalar fallback.\n");
        }
        else {
            lcg_fill  = lcg_fill_block_avx2;
            lcg_check = lcg_check_block_avx2;
        }
    }
  #endif
#endif /* HAVE_X86 */
}

/* a list of open file handles */
int* g_filehandle = NULL;
unsigned int g_filehandle_size = 0;
unsigned int g_filehandle_limit = 0;

/* append to the list of open file handles */
void filehandle_append(int fd)
{
    if (g_filehandle_size >= g_filehandle_limit)
    {
        int* new_filehandle;

        g_filehandle_limit *= 2;
        if (g_filehandle_limit < 128) g_filehandle_limit = 128;

        new_filehandle = realloc(
            g_filehandle, sizeof(int) * g_filehandle_limit);
        if (!new_filehandle) {
            fprintf(stderr,
                    "Out of memory when allocating new file handle buffer.\n");
            exit(EXIT_FAILURE);
        }
        g_filehandle = new_filehandle;
    }

    g_filehandle[ g_filehandle_size++ ] = fd;
}

/* produce nicely formatted time in seconds */
void format_time(unsigned int sec, char output[64])
{
    /* maximum digits of 32-bit unsigned int are 9. */
    if (sec >= 24 * 3600) {
        unsigned int days = sec / (24 * 3600);
        sec -= days * (24 * 3600);
        unsigned int hours = sec / 3600;
        sec -= hours * 3600;
        unsigned int minutes = sec / 60;
        sec -= minutes * 60;
        sprintf(output, "%ud%uh%um%us", days, hours, minutes, sec);
    }
    else if (sec >= 3600) {
        unsigned int hours = sec / 3600;
        sec -= hours * 3600;
        unsigned int minutes = sec / 60;
        sec -= minutes * 60;
        sprintf(output, "%uh%um%us", hours, minutes, sec);
    }
    else if (sec >= 60) {
        unsigned int minutes = sec / 60;
        sec -= minutes * 60;
        sprintf(output, "%um%us", minutes, sec);
    }
    else {
        sprintf(output, "%us", sec);
    }
}

/* for compatibility with windows, use O_BINARY if available */
#ifndef O_BINARY
#define O_BINARY 0
#endif

/* print command line usage */
void print_usage(char* argv[])
{
    fprintf(stderr,
            "Usage: %s [-s seed] [-f files] [-S size] [-r] [-u] [-U] [-C dir] [-t seconds] [-T seconds]\n"
            "\n"
            "disk-filltest " VERSION " is a simple program which fills a path with random\n"
            "data and then rereads the files to check that the random sequence was\n"
            "correctly stored. Supports AVX2 acceleration on x86/x86_64 CPUs.\n"
            "\n"
            "Options: \n"
            "  -C <dir>          Change into given directory before starting work.\n"
            "  -f <file number>  Only write this number of 1 GiB sized files.\n"
            "  -N                Skip verification, e.g. for just wiping a disk.\n"
            "  -r                Only verify existing data files with given random seed.\n"
            "  -R <times>        Repeat fill/test/wipe steps given number of times.\n"
            "  -s <random seed>  Use random seed to write or verify data files.\n"
            "  -S <size>         Size of each random file in MiB (default: 1024).\n"
            "  -t <seconds>      Delay in seconds after each file (except the last).\n"
            "  -T <seconds>      Delay in seconds between -R repetitions (except after the last).\n"
            "  -u                Remove files after successful test.\n"
            "  -U                Immediately remove files, write and verify via file handles.\n"
            "  -V                Print version and exit.\n"
            "\n",
            argv[0]);
    exit(EXIT_FAILURE);
}

/* parse command line parameters */
void parse_commandline(int argc, char* argv[])
{
    int opt;

    while ((opt = getopt(argc, argv, "hs:S:f:ruUC:NR:Vt:T:")) != -1) {
        switch (opt) {
        case 's':
            g_seed = atoi(optarg);
            break;
        case 'S':
            gopt_file_size = atoi(optarg);
            break;
        case 'f':
            gopt_file_limit = atoi(optarg);
            break;
        case 'r':
            gopt_readonly = 1;
            break;
        case 'u':
            gopt_unlink_after = 1;
            break;
        case 'U':
            gopt_unlink_immediate = 1;
            break;
        case 'C':
            if (chdir(optarg) != 0) {
                printf("Error chdir to %s: %s\n", optarg, strerror(errno));
                exit(EXIT_FAILURE);
            }
            break;
        case 'N':
            gopt_skip_verify = 1;
            break;
        case 'R':
            gopt_repeat = atoi(optarg);
            break;
	case 'V':
	    printf("disk-filltest " VERSION "\n");
            exit(EXIT_SUCCESS);
        case 't':
            gopt_delay = atoi(optarg);
            break;
        case 'T':
            gopt_repeat_delay = atoi(optarg);
            break;
        case 'h':
        default:
            print_usage(argv);
        }
    }

    if (optind < argc)
        print_usage(argv);

    if (gopt_file_size == 0)
        gopt_file_size = 1024;
}

/* unlink random files; label describes what is being removed (e.g. "old
 * files" before a run, "test files" after a successful run). */
void unlink_randfiles(const char* label)
{
    unsigned int filenum = 0;
    int is_tty = isatty(fileno(stdout));

    while (filenum < UINT_MAX)
    {
        char filename[32];
        sprintf(filename, "random-%08u", filenum);

        if (unlink(filename) != 0)
            break;

        ++filenum;

        if (is_tty) {
            printf("\rRemoving %s: %u", label, filenum);
            fflush(stdout);
        }
    }

    if (filenum > 0) {
        if (is_tty)
            printf("\rRemoving %s: %u, done.\n", label, filenum);
        else
            printf("Removed %s: %u.\n", label, filenum);
    }
}

/* fill disk */
void write_randfiles(void)
{
    unsigned int filenum = 0;
    int done = 0;
    unsigned int expected_file_limit = UINT_MAX;

    if (gopt_file_limit == UINT_MAX) {
#if HAVE_STATVFS
        struct statvfs buf;

        if (statvfs(".", &buf) == 0) {
            uint64_t free_size =
                (uint64_t)(buf.f_blocks) * (uint64_t)(buf.f_bsize);

            expected_file_limit = (free_size + gopt_file_size - 1)
                / (1024 * 1024) / gopt_file_size;
        }
#endif /* HAVE_STATVFS */
    }
    else {
        expected_file_limit = gopt_file_limit;
    }

    printf("Writing files random-######## with seed %u\n", g_seed);

    while (!done && filenum < gopt_file_limit)
    {
        char filename[32], eta[64];
        int fd;
        ssize_t wb;
        unsigned int blocknum, wp;
        uint64_t wtotal;
        double ts1, ts2, speed;
        uint64_t rnd;

        item_type block[(1024 * 1024) / sizeof(item_type)];

        sprintf(filename, "random-%08u", filenum);

        fd = open(filename, O_RDWR | O_CREAT | O_TRUNC | O_BINARY, 0600);
        if (fd < 0) {
            printf("Error opening next file %s: %s\n",
                   filename, strerror(errno));
            break;
        }

        if (gopt_unlink_immediate) {
            if (unlink(filename) != 0) {
                printf("Error unlinking opened file %s: %s\n",
                       filename, strerror(errno));
            }
        }

        /* reset random generator for each 1 GiB file */
        rnd = g_seed + (++filenum);

        wtotal = 0;
        ts1 = timestamp();

        for (blocknum = 0; blocknum < gopt_file_size; ++blocknum)
        {
            rnd = lcg_fill(block,
                           sizeof(block) / sizeof(item_type), rnd);

            wp = 0;

            while ( wp != sizeof(block) && !done )
            {
                wb = write(fd, (char*)block + wp, sizeof(block) - wp);

                if (wb <= 0) {
                    printf("Error writing next file %s: %s\n",
                           filename, strerror(errno));
                    done = 1;
                    break;
                }
                else {
                    wp += wb;
                }
            }

            wtotal += wp;
        }

        if (gopt_unlink_immediate) { /* do not close file handle! */
            filehandle_append(fd);
        }
        else {
            close(fd);
        }

        ts2 = timestamp();

        speed = wtotal / 1024.0 / 1024.0 / (ts2 - ts1);
        g_last_filesize = wtotal;

        if (expected_file_limit != UINT_MAX && filenum <= expected_file_limit) {
            format_time(
                (expected_file_limit - filenum) * gopt_file_size / speed, eta);

            printf("Wrote %.0f MiB random data to %s with %f MiB/s, eta %s.\n",
                   (wtotal / 1024.0 / 1024.0), filename, speed, eta);
        }
        else {
            printf("Wrote %.0f MiB random data to %s with %f MiB/s.\n",
                   (wtotal / 1024.0 / 1024.0), filename, speed);
        }
        fflush(stdout);

        /* add delay after each file (except the last) if requested */
        if (gopt_delay > 0 && gopt_file_limit > 1 && filenum < gopt_file_limit) {
            printf("Sleeping for %d seconds...\n", gopt_delay);
            fflush(stdout);
            sleep(gopt_delay);
        }
    }

    errno = 0;
}

/* read files and check random sequence*/
void read_randfiles(void)
{
    unsigned int filenum = 0;
    int done = 0;
    unsigned int expected_file_limit = UINT_MAX;

    if (gopt_unlink_immediate) {
        expected_file_limit = g_filehandle_size;
    }
    else {
        char filename[32];
        int fd;

        for (expected_file_limit = 0; ; ++expected_file_limit) {
            /* attempt to open file */
            sprintf(filename, "random-%08u", expected_file_limit);
            fd = open(filename, O_RDONLY | O_BINARY);
            if (fd < 0)
                break;
            close(fd);
        }
    }

    printf("Verifying %u files random-######## with seed %u\n",
           expected_file_limit, g_seed);

    while (!done)
    {
        char filename[32], eta[64];
        int fd;
        ssize_t rb;
        unsigned int blocknum;
        uint64_t rtotal;
        double ts1, ts2, speed;
        uint64_t rnd;

        item_type block[(1024 * 1024) / sizeof(item_type)];

        sprintf(filename, "random-%08u", filenum);

        if (gopt_unlink_immediate)
        {
            if (filenum >= g_filehandle_size) {
                printf("Finished all opened file handles.\n");
                break;
            }

            fd = g_filehandle[filenum];

            if (lseek(fd, 0, SEEK_SET) != 0) {
                printf("Error seeking in next file %s: %s\n",
                       filename, strerror(errno));
                exit(EXIT_FAILURE);
            }
        }
        else
        {
            fd = open(filename, O_RDONLY | O_BINARY);
            if (fd < 0) {
                printf("Error opening next file %s: %s\n",
                       filename, strerror(errno));
                break;
            }
        }

        /* reset random generator for each 1 GiB file */
        rnd = g_seed + (++filenum);

        rtotal = 0;
        ts1 = timestamp();

        for (blocknum = 0; blocknum < gopt_file_size; ++blocknum)
        {
            unsigned int read_size = sizeof(block);
            if (filenum == expected_file_limit && g_last_filesize != UINT_MAX &&
                blocknum * sizeof(block) > g_last_filesize) {
                read_size = g_last_filesize - (blocknum - 1) * sizeof(block);
            }
            rb = read(fd, block, read_size);

            if (rb == 0) {
                /* got EOF on file */
                if (filenum != expected_file_limit ||
                    (g_last_filesize != UINT_MAX && rtotal != g_last_filesize))
                {
                    printf("Unexpectedly short file %s: "
                           "read %u of expected %"PRIu64" bytes\n",
                           filename, g_last_filesize, rtotal);
                    done = 1;
                    exit(EXIT_FAILURE);
                }

                done = 1;
                break;
            }
            else if (rb < 0) {
                printf("Error reading file %s: %s\n",
                       filename, strerror(errno));
                done = 1;
                exit(EXIT_FAILURE);
            }

            {
                unsigned int elem_count = rb / sizeof(item_type);
                unsigned int mismatch = lcg_check(
                    block, elem_count, rnd, &rnd);
                if (mismatch)
                {
                    printf("Mismatch to random sequence "
                           "in file %s block %d at offset %lu\n",
                           filename, blocknum,
                           (long unsigned)((mismatch - 1) * sizeof(int)));
                    gopt_unlink_after = 0;
                    exit(EXIT_FAILURE);
                }
            }

            rtotal += rb;
        }

        close(fd);

        ts2 = timestamp();

        speed = rtotal / 1024.0 / 1024.0 / (ts2 - ts1);
        format_time(
            (expected_file_limit - filenum) * gopt_file_size / speed, eta);

        printf("Read %.0f MiB random data from %s with %f MiB/s, eta %s.\n",
               (rtotal / 1024.0 / 1024.0), filename, speed, eta);
        fflush(stdout);
    }

    printf("Successfully verified %u files random-######## with seed %u\n",
           expected_file_limit, g_seed);
}

int main(int argc, char* argv[])
{
    int r;

    g_seed = time(NULL);

    parse_commandline(argc, argv);

    lcg_init();

    for (r = 0; r < gopt_repeat; ++r)
    {
        if (gopt_readonly)
        {
            read_randfiles();
            if (gopt_unlink_after)
                unlink_randfiles("test files");
        }
        else
        {
            unlink_randfiles("old files");
            write_randfiles();
            if (!gopt_skip_verify)
                read_randfiles();
            if (gopt_unlink_after)
                unlink_randfiles("test files");
        }

        if (gopt_repeat_delay > 0 && r + 1 < gopt_repeat) {
            printf("Sleeping for %d seconds before next repetition...\n",
                   gopt_repeat_delay);
            fflush(stdout);
            sleep(gopt_repeat_delay);
        }
    }

    return 0;
}

/******************************************************************************/
