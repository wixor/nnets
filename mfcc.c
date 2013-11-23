#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

#if 0
static int sample_rate = 16000; /* 16 kHz */
static int frame_length = 400; /* 25 ms */
static int frame_spacing = 160; /* 10 ms */
sattic int mel_filters = 26;
#endif
static int sample_rate = 16000;
static int frame_length = 400; /* 25 ms */
static int frame_spacing = 160; /* 10 ms */
static int mel_filters = 26;
static float mel_high_freq = 8000.f; /* Hz */

typedef int16_t sample_t;

static inline float sample_to_float(sample_t s) {
    return (float)s / 32768.f;
}

static inline float hz_to_mel(float hz) {
    return 1125.f * log1pf(hz / 700.f);
}
static inline float mel_to_hz(float mel) {
    return 700.f * (expf(mel/1125.f) - 1.f);
}

static bool read_frame(sample_t *rdbuf, bool first)
{
    if(!first)
        memmove(rdbuf, rdbuf+frame_spacing, sizeof(sample_t) * (frame_length - frame_spacing));

    sample_t *bufend = rdbuf + frame_length,
             *wrptr = first ? rdbuf : bufend - frame_spacing;

    bool success = false;
    while(wrptr < bufend)
    {
        int rdcnt = fread(wrptr, sizeof(sample_t), bufend - wrptr, stdin);
        if(feof(stdin))
            break;
        if(ferror(stdin)) {
            perror("failed to read from stdin");
            exit(EXIT_FAILURE);
        }
        wrptr += rdcnt;
        success = true;
    }

    memset(wrptr, 0, sizeof(sample_t) * (bufend - wrptr));
    return success;
}

int main(void)
{
    sample_t *rdbuf = malloc(sizeof(sample_t) * frame_length);

    fftw_complex *fft_in = fftw_alloc_complex(frame_length),
                 *fft_out = fftw_alloc_complex(frame_length);

    fftw_plan freq_plan = fftw_plan_dft_1d(frame_length, fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);

    float *window = malloc(sizeof(float) * frame_length),
          *mel_freqs = malloc(sizeof(float) * (mel_filters+2)),
          *mel_power = malloc(sizeof(float) * mel_filters);

    for(int i=0; i<frame_length; i++)
        window[i] = .5f - .5f*cosf((float)(2. * M_PI) * i / (frame_length-1));

    float mel_step = hz_to_mel(mel_high_freq) / (mel_filters+1);
    for(int i=0; i<mel_filters+2; i++)
        mel_freqs[i] = mel_to_hz(mel_step * i);

    fprintf(stderr,
            "sample rate %d Hz; frame length: %d samples, frame spacing: %d samples\n"
            "%d mel filters:", sample_rate, frame_length, frame_spacing, mel_filters);
    for(int i=0; i<mel_filters+2; i++)
        fprintf(stderr, " %.1f Hz", mel_freqs[i]);
    fputc('\n', stderr);

    fwrite(&mel_filters, sizeof(int), 1, stdout);

    int frame_count = 0;
    while(read_frame(rdbuf, 0 == frame_count))
    {
        frame_count += 1;

        for(int i=0; i<frame_length; i++)
            fft_in[i] = sample_to_float(rdbuf[i]) * window[i];

        fftw_execute(freq_plan);
        
        for(int j=0; j<mel_filters; j++)
        {
            float lo = mel_freqs[j],
                  mid = mel_freqs[j+1],
                  high = mel_freqs[j+2];

            float accum = 0;

            for(int i=0; i<frame_length/2; i++)
            {
                float freq = (float)(sample_rate * i) / frame_length,
                      re = crealf(fft_out[i]) / frame_length,
                      im = cimagf(fft_out[i]) / frame_length,
                      power = re*re + im*im;

                if(i != 0) power *= 2.f;

                if(freq > lo && freq < high)
                    accum += (freq < mid)
                                ? (freq-lo) / (mid-lo) * power
                                : (high-freq) / (high-mid) * power;

            }

            mel_power[j] = accum;
        }

        fwrite(mel_power, sizeof(float)*mel_filters, 1, stdout);
    }
    
    fprintf(stderr, "processed %d frames\n", frame_count);

    free(mel_power);
    free(mel_freqs);
    free(window);

    fftw_destroy_plan(freq_plan);

    fftw_free(fft_out);
    fftw_free(fft_in);
    free(rdbuf);

    return 0;
}
