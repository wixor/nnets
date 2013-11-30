#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <fftw3.h>

static int sample_rate = 16000; /* 16 kHz */
static int frame_length = 400; /* 25 ms */
static int frame_spacing = 160; /* 10 ms */

static int mel_filters = 26;
static float mel_high_freq = 8000.f; /* Hz */
static float mel_power_threshold = -70.f; /* dB */

static int dct_length = 13; /* mel filters / 2 */

typedef int16_t sample_t;

struct {
    char *buf, *rdptr, *wrptr, *endptr;
} inbuf;

static fftw_complex *fft_in, *fft_out;
static fftw_plan fft_plan;

static double *dct_in, *dct_out;
static fftw_plan dct_plan;

static int fft_length;
static float *window;
static float *fft_freqs, *fft_power;
static float *mel_freqs, *mel_power;
static float *dct_coeffs;

static inline float sample_to_float(sample_t s) {
    return (float)s / 32768.f;
}
static inline float hz_to_mel(float hz) {
    return 1125.f * log1pf(hz / 700.f);
}
static inline float mel_to_hz(float mel) {
    return 700.f * (expf(mel/1125.f) - 1.f);
}
static inline float power_to_db(float p) {
    return logf(p) * 4.3429448f; /* decibels */
}
static inline float db_to_power(float p) {
    return expf(0.23025851f * p);
}

static void process_frame(const sample_t *samples)
{
    for(int i=0; i<frame_length; i++)
        fft_in[i] = sample_to_float(samples[i]) * window[i];

    fftw_execute(fft_plan);
    
    for(int i=0; i<fft_length; i++)
    {
        float re = crealf(fft_out[i]) / frame_length,
              im = cimagf(fft_out[i]) / frame_length,
              power = re*re + im*im;
        if(i != 0) power *= 2.f;
        fft_power[i] = power;
    }

    float min_mel_power = db_to_power(mel_power_threshold);

    for(int j=0; j<mel_filters; j++)
    {
        float lo = mel_freqs[j],
              mid = mel_freqs[j+1],
              high = mel_freqs[j+2];

        float accum = 0;
        
        for(int i=0; i<fft_length; i++)
        {
            float freq = fft_freqs[i];
            if(freq <= lo || freq >= high)
                continue;

            accum += fft_power[i] *
                ((freq < mid)  ?  (freq-lo) / (mid-lo) :  (high-freq) / (high-mid));
        }
        
        dct_in[j] = mel_power[j] = power_to_db(accum + min_mel_power);
    }

    for(int i=0; i<fft_length; i++)
        fft_power[i] = power_to_db(fft_power[i]);

    fftw_execute(dct_plan);
    for(int i=0; i<dct_length; i++)
        dct_coeffs[i] = (float)(dct_out[i] / (2*mel_filters));
}

static void initialize()
{
    inbuf.buf = malloc(sizeof(sample_t) * 4096);
    inbuf.rdptr = inbuf.wrptr = inbuf.buf;
    inbuf.endptr = inbuf.buf + sizeof(sample_t) * 4096;

    fft_in = fftw_alloc_complex(frame_length);
    fft_out = fftw_alloc_complex(frame_length);

    dct_in = malloc(sizeof(double) * mel_filters);
    dct_out = malloc(sizeof(double) * mel_filters);

    fft_plan = fftw_plan_dft_1d(frame_length, fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
    dct_plan = fftw_plan_r2r_1d(mel_filters, dct_in, dct_out, FFTW_REDFT10, FFTW_MEASURE);

    window = malloc(sizeof(float) * frame_length);
    
    fft_length = frame_length / 2;
    fft_freqs = malloc(sizeof(float) * fft_length);
    fft_power = malloc(sizeof(float) * fft_length);

    mel_freqs = malloc(sizeof(float) * (mel_filters+2));
    mel_power = malloc(sizeof(float) * mel_filters);
   
    dct_length = mel_filters / 2; 
    dct_coeffs = malloc(sizeof(float) * dct_length);

    /* Hann's window */
    for(int i=0; i<frame_length; i++)
        window[i] = .5f - .5f*cosf((float)(2. * M_PI) * i / frame_length);
    
    for(int i=0; i<fft_length; i++)
        fft_freqs[i] = (float)(sample_rate * i) / frame_length;

    float mel_step = hz_to_mel(mel_high_freq) / (mel_filters+1);
    for(int i=0; i<mel_filters+2; i++)
        mel_freqs[i] = mel_to_hz(mel_step * i);
}

static void say_hello()
{
    char buf[512], *ptr = buf;

    ptr += sprintf(ptr,
            "sample rate %d Hz; frame length: %d samples, frame spacing: %d samples\n"
            "%d mel filters:", sample_rate, frame_length, frame_spacing, mel_filters);
    for(int i=0; i<mel_filters+2; i++)
        ptr += sprintf(ptr, " %.1f Hz", mel_freqs[i]);
    ptr += sprintf(ptr, "\n%d dct coefficients\n", dct_length);

    fputs(buf, stderr);
}

static void cleanup()
{
    fflush(stdout);

    free(dct_coeffs);

    free(mel_power);
    free(mel_freqs);

    free(fft_power);
    free(fft_freqs);

    free(window);

    fftw_destroy_plan(dct_plan);
    fftw_destroy_plan(fft_plan);

    free(dct_out);
    free(dct_in);

    fftw_free(fft_out);
    fftw_free(fft_in);

    free(inbuf.buf);
}

static int get_stdin_flags() {
    int ret = fcntl(STDIN_FILENO, F_GETFL);
    if(-1 == ret) {
        perror("fcntl(F_GETFL)");
        exit(EXIT_FAILURE);
    }
    return ret;
}

static void set_stdin_flags(int fl) {
    if(-1 == fcntl(STDIN_FILENO, F_SETFL, fl)) {
        perror("fcntl(F_SETFL)");
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    initialize();
    say_hello();

    fwrite(&mel_filters, sizeof(int), 1, stdout);
    fwrite(&fft_length,  sizeof(int), 1, stdout);
    fwrite(&dct_length,  sizeof(int), 1, stdout);

    fwrite(mel_freqs, sizeof(float)*mel_filters, 1, stdout);
    fwrite(fft_freqs, sizeof(float)*fft_length, 1, stdout);

    int stdin_flags = get_stdin_flags();
    stdin_flags |= O_NONBLOCK;
    set_stdin_flags(stdin_flags);

    const int frame_bytes = sizeof(sample_t) * frame_length,
              frame_offs = sizeof(sample_t) * frame_spacing;

    int frame_count = 0;

    for(bool eof = false; !eof; )
    {
        int rc;

        if(inbuf.endptr - inbuf.rdptr < frame_bytes)
        {
            memmove(inbuf.buf, inbuf.rdptr, inbuf.wrptr - inbuf.rdptr);
            inbuf.wrptr = inbuf.buf + (inbuf.wrptr - inbuf.rdptr);
            inbuf.rdptr = inbuf.buf;
        }

        while(inbuf.wrptr - inbuf.rdptr < frame_bytes)
        {
            rc = read(STDIN_FILENO, inbuf.wrptr, inbuf.endptr - inbuf.wrptr);

            if(0 == rc) {
                memset(inbuf.wrptr, 0, inbuf.endptr - inbuf.wrptr);
                eof = true;
                break;
            }

            if(-1 == rc)
            {
                if(EAGAIN == errno) {
                    fflush(stdout);
                    set_stdin_flags(stdin_flags &= ~O_NONBLOCK);
                    continue;
                }
                perror("read");
                return EXIT_FAILURE;
            }

            inbuf.wrptr += rc;
        }

        if(!(stdin_flags & O_NONBLOCK))
            set_stdin_flags(stdin_flags |= O_NONBLOCK);

        process_frame((const sample_t *)inbuf.rdptr);
        fwrite(mel_power,  sizeof(float)*mel_filters, 1, stdout);
        fwrite(fft_power,  sizeof(float)*fft_length, 1, stdout);
        fwrite(dct_coeffs, sizeof(float)*dct_length, 1, stdout);
        
        frame_count += 1;
        inbuf.rdptr += frame_offs;
    }
    
    fprintf(stderr, "processed %d frames\n", frame_count);

    cleanup();

    return 0;
}
