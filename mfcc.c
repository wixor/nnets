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

typedef int16_t sample_t;

struct {
    char *buf, *rdptr, *wrptr, *endptr;
} inbuf;

static fftw_complex *fft_in, *fft_out;
static fftw_plan fft_plan;

static float *window;
static float *fft_freqs, *fft_power;
static float *mel_freqs, *mel_power;

static inline float sample_to_float(sample_t s) {
    return (float)s / 32768.f;
}
static inline float hz_to_mel(float hz) {
    return 1125.f * log1pf(hz / 700.f);
}
static inline float mel_to_hz(float mel) {
    return 700.f * (expf(mel/1125.f) - 1.f);
}

static void process_frame(const sample_t *samples)
{
    for(int i=0; i<frame_length; i++)
        fft_in[i] = sample_to_float(samples[i]) * window[i];

    fftw_execute(fft_plan);
    
    for(int i=0; i<frame_length/2; i++)
    {
        float re = crealf(fft_out[i]) / frame_length,
              im = cimagf(fft_out[i]) / frame_length,
              power = re*re + im*im;
        if(i != 0) power *= 2.f;
        fft_power[i] = logf(power) / 0.2302585092994046f; /* decibels */
    }

    for(int j=0; j<mel_filters; j++)
    {
        float lo = mel_freqs[j],
              mid = mel_freqs[j+1],
              high = mel_freqs[j+2];

        float num = 0, denom = 0;
        
        for(int i=0; i<frame_length/2; i++)
        {
            float freq = fft_freqs[i];
            if(freq <= lo || freq >= high)
                continue;

            float weight =
                (freq < mid) ? (freq-lo) / (mid-lo) : (high-freq) / (high-mid);
            num += weight * fft_power[i];
            denom += weight;
        }

        mel_power[j] = num / denom;
    }
}

static void initialize()
{
    inbuf.buf = malloc(sizeof(sample_t) * 4096);
    inbuf.rdptr = inbuf.wrptr = inbuf.buf;
    inbuf.endptr = inbuf.buf + sizeof(sample_t) * 4096;

    fft_in = fftw_alloc_complex(frame_length);
    fft_out = fftw_alloc_complex(frame_length);

    fft_plan = fftw_plan_dft_1d(frame_length, fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);

    window = malloc(sizeof(float) * frame_length);
    
    fft_freqs = malloc(sizeof(float) * (frame_length/2));
    fft_power = malloc(sizeof(float) * (frame_length/2));

    mel_freqs = malloc(sizeof(float) * (mel_filters+2));
    mel_power = malloc(sizeof(float) * mel_filters);

    for(int i=0; i<frame_length; i++)
        window[i] = .5f - .5f*cosf((float)(2. * M_PI) * i / (frame_length-1));
    
    for(int i=0; i<frame_length/2; i++)
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
    ptr += sprintf(ptr, "\n");

    fputs(buf, stderr);
}

static void cleanup()
{
    fflush(stdout);

    free(mel_power);
    free(mel_freqs);

    free(fft_power);
    free(fft_freqs);

    free(window);

    fftw_destroy_plan(fft_plan);
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

    int n_fft_freqs = frame_length/2;
    fwrite(&mel_filters, sizeof(int), 1, stdout);
    fwrite(&n_fft_freqs, sizeof(int), 1, stdout);

    fwrite(mel_freqs, sizeof(float)*mel_filters, 1, stdout);
    fwrite(fft_freqs, sizeof(float)*n_fft_freqs, 1, stdout);

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
        fwrite(mel_power, sizeof(float)*mel_filters, 1, stdout);
        fwrite(fft_power, sizeof(float)*(frame_length/2), 1, stdout);
        
        frame_count += 1;
        inbuf.rdptr += frame_offs;
    }
    
    fprintf(stderr, "processed %d frames\n", frame_count);

    cleanup();

    return 0;
}
