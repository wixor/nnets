#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>

#include <fftw3.h>
#include <vorbis/vorbisfile.h>

static struct {
    int streamer_buffer;
    float frame_sec;
    float step_sec;
    int mel_filters;
    float mel_high_freq;
    float mel_power_threshold;
} config = {
    .streamer_buffer = 8192,
    .frame_sec = 0.015f,
    .step_sec = 0.005f,
    .mel_filters = 21,
    .mel_high_freq = 4270.f,
    .mel_power_threshold = -70.f,
};

/* ------------------------------------------------------------------------- */

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
static inline float power_to_db(float p) {
    return logf(p) * 4.3429448f; /* decibels */
}
static inline float db_to_power(float p) {
    return expf(0.23025851f * p);
}

/* ------------------------------------------------------------------------- */

#define WAVELET_BORDER 6

static inline float wavelet_predict(float a, float b, float c, float d) {
    /* return (9.f * (a + b) - (c + d)) * 0.0625; */
    return .5f * (a + b);
}
static inline float wavelet_update(float a, float b, float c, float d) {
    return .25f * (a + b);
}

static void wavelet_mirror(float *aux, int n)
{
    aux[-1] = aux[1];
    aux[n] = aux[n-2];
    aux[-2] = aux[2];
    aux[n+1] = aux[n-3];
    aux[-3] = aux[3];
    aux[n+2] = aux[n-4];
}

static inline void wavelet_forward_step(float *buf, float *aux, int n)
{
    memcpy(aux, buf, n*sizeof(float));
    
    wavelet_mirror(aux, n);
    for(int i=1; i<n; i+=2)
        aux[i] -= wavelet_predict(aux[i-1], aux[i+1], aux[i-3], aux[i+3]);
    
    wavelet_mirror(aux, n);
    for(int i=0; i<n; i+=2)
        aux[i] += wavelet_update(aux[i-1], aux[i+1], aux[i-3], aux[i+3]);
    
    for(int i=0; 2*i<n; i++)
        buf[i] = aux[2*i];
    for(int i=0; 2*i+1<n; i++)
        buf[i + (n+1)/2] = aux[2*i+1];
}
static inline void wavelet_backward_step(float *buf, float *aux, int n)
{
    for(int i=0; 2*i<n; i++)
        aux[2*i] = buf[i];
    for(int i=0; 2*i+1<n; i++)
        aux[2*i+1] = buf[i + (n+1)/2];

    wavelet_mirror(aux, n);
    for(int i=0; i<n; i+=2)
        aux[i] -= wavelet_update(aux[i-1], aux[i+1], aux[i-3], aux[i+3]);
    
    wavelet_mirror(aux, n);
    for(int i=1; i<n; i+=2)
        aux[i] += wavelet_predict(aux[i-1], aux[i+1], aux[i-3], aux[i+3]);
   
    memcpy(buf, aux, sizeof(float)*n); 
}

static void wavelet_forward(float *buf, float *aux, int n) {
    if(n < 2)
        return;
    wavelet_forward_step(buf, aux+(WAVELET_BORDER/2), n);
    wavelet_forward(buf, aux, (n+1)/2);
}
static void wavelet_backward(float *buf, float *aux, int n) {
    if(n < 2)
        return;
    wavelet_backward(buf, aux, (n+1)/2);
    wavelet_backward_step(buf, aux+(WAVELET_BORDER/2), n);
}

/* ------------------------------------------------------------------------- */

class mfcc
{
public:
    struct profile
    {
        int sample_rate;
        int frame_length;
        int frame_spacing;
        int num_channels;
        int mel_filters;
        float mel_high_freq;
        float mel_power_threshold;
        
        bool operator==(const struct profile &p) const
        {
            return sample_rate == p.sample_rate &&
                   frame_length == p.frame_length &&
                   frame_spacing == p.frame_spacing &&
                   num_channels == p.num_channels &&
                   mel_filters == p.mel_filters &&
                   mel_high_freq == p.mel_high_freq &&
                   mel_power_threshold == p.mel_power_threshold;
        }
        
        inline bool operator!=(const profile &p) const {
            return !((*this) == p);
        }
    };

    struct profile p;

    float mel_power_offs;

    int fft_length;
    fftw_complex *fft_in, *fft_out;
    fftw_plan fft_plan;

    double *dct_in, *dct_out;
    fftw_plan dct_plan;

    float *window;
    float *fft_freqs, *fft_power;
    float *mel_freqs, *mel_power;
    float *dct_coeffs;
    float *wvl_coeffs, *wvl_aux;

    mfcc(const struct mfcc::profile &_p);
    ~mfcc();
    void process_frame(const sample_t *samples);
};

mfcc::mfcc(const struct mfcc::profile &_p)
{
    p = _p;

    mel_power_offs = db_to_power(p.mel_power_threshold);

    fft_in = fftw_alloc_complex(p.frame_length);
    fft_out = fftw_alloc_complex(p.frame_length);

    dct_in = (double *)malloc(sizeof(double) * p.mel_filters);
    dct_out = (double *)malloc(sizeof(double) * p.mel_filters);

    fft_plan = fftw_plan_dft_1d(p.frame_length, fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
    dct_plan = fftw_plan_r2r_1d(p.mel_filters, dct_in, dct_out, FFTW_REDFT10, FFTW_MEASURE);

    window = (float *)malloc(sizeof(float) * p.frame_length);
    
    fft_length = p.frame_length / 2;
    fft_freqs = (float *)malloc(sizeof(float) * fft_length);
    fft_power = (float *)malloc(sizeof(float) * fft_length);

    mel_freqs = (float *)malloc(sizeof(float) * (p.mel_filters+2));
    mel_power = (float *)malloc(sizeof(float) * p.mel_filters);
   
    dct_coeffs = (float *)malloc(sizeof(float) * p.mel_filters);
    
    wvl_coeffs = (float *)malloc(sizeof(float) * p.mel_filters);
    wvl_aux = (float *)malloc(sizeof(float) * (p.mel_filters + WAVELET_BORDER));

    /* Hann's window */
    for(int i=0; i<p.frame_length; i++)
        window[i] = .5f - .5f*cosf((float)(2. * M_PI) * i / p.frame_length);
    
    for(int i=0; i<fft_length; i++)
        fft_freqs[i] = (float)(p.sample_rate * i) / p.frame_length;

    mel_power_offs = db_to_power(p.mel_power_threshold);
    float mel_step = hz_to_mel(p.mel_high_freq) / (p.mel_filters+1);
    for(int i=0; i<p.mel_filters+2; i++)
        mel_freqs[i] = mel_to_hz(mel_step * i);
}

mfcc::~mfcc()
{
    free(wvl_aux);
    free(wvl_coeffs);

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
}


void mfcc::process_frame(const sample_t *samples)
{
    if(1 == p.num_channels)
        for(int i=0; i<p.frame_length; i++)
            fft_in[i] = sample_to_float(samples[i]) * window[i];
    else
        for(int i=0; i<p.frame_length; i++)
            fft_in[i] = .5f * (sample_to_float(samples[2*i]) + sample_to_float(samples[2*i+1])) * window[i];

    fftw_execute(fft_plan);
    
    for(int i=0; i<fft_length; i++)
    {
        float re = crealf(fft_out[i]) / p.frame_length,
              im = cimagf(fft_out[i]) / p.frame_length,
              power = re*re + im*im;
        if(i != 0) power *= 2.f;
        fft_power[i] = power;
    }

    for(int j=0; j<p.mel_filters; j++)
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
        
        dct_in[j] = mel_power[j] = power_to_db(accum + mel_power_offs);
    }

    for(int i=0; i<fft_length; i++)
        fft_power[i] = power_to_db(fft_power[i]);

    fftw_execute(dct_plan);
    for(int i=0; i<p.mel_filters; i++)
        dct_coeffs[i] = dct_out[i] / (2*p.mel_filters);

    memcpy(wvl_coeffs, mel_power, sizeof(float)*p.mel_filters);
    wavelet_forward(wvl_coeffs, wvl_aux, p.mel_filters);
}

/* ------------------------------------------------------------------------- */

struct lookahead
{
    char data[8];
    int size;

    lookahead(int fd);

    inline bool match(const char *pat) const {
        return match(pat, strlen(pat));
    }
    inline bool match(const char *pat, int n) const {
        return size >= n && 0 == memcmp(data, pat, n);
    }
};

lookahead::lookahead(int fd) 
{
    ssize_t rc = read(fd, data, sizeof(data));
    if(-1 == rc) {
        perror("read");
        exit(EXIT_FAILURE);
    }
    size = rc;
}

/* ------------------------------------------------------------------------- */

class streamer
{
protected:
    int fd;
    char *buf, *rdptr, *wrptr, *endptr;
    int num_channels, sample_rate, bits_per_sample;
    int sample_offset;

    streamer(int _fd, int buffer_size);
    void need_buffer_space(int frame_bytes);
    void check_format();

public:
    enum read_status {
        READ_OK, READ_STALL, READ_EOF
    };

    virtual ~streamer() = 0;
    virtual enum read_status read(int samples) = 0;
    virtual void shutdown() = 0;

    inline const sample_t *get_samples() const { return (const sample_t *)rdptr; }
    inline int get_sample_rate() const { return sample_rate; }
    inline int get_num_channels() const { return num_channels; }
    inline int get_sample_offset() const { return sample_offset; }

    inline void advance(int step) {
        rdptr += sizeof(sample_t) * num_channels * step;
        sample_offset += step;
    }

    void make_profile(struct mfcc::profile *profile) const;
};

class wav_streamer : public streamer
{
protected:
    int fd_flags;

    void parse_wav(const struct lookahead &la);
    void assume_raw(const struct lookahead &la);
    void set_blocking(bool blocking);

    struct wavhdr {
        char riff_chunkid[4];
        int32_t riff_chunksize;
        char riff_format[4];

        char sub1_chunkid[4];
        int32_t sub1_chunksize;
        int16_t audiofmt;
        int16_t num_channels;
        int32_t sample_rate;
        int32_t byte_align;
        int16_t block_align;
        int16_t bits_per_sample;

        char sub2_chunkid[4];
        int32_t sub2_chunksize;
    };

public:
    wav_streamer(int _fd, int buffer_size, const struct lookahead &la);
    virtual ~wav_streamer();
    virtual enum read_status read(int samples);
    virtual void shutdown();
};

class vorbis_streamer : public streamer
{
protected:
    static size_t read_callback(void *ptr, size_t size, size_t nmemb, void *datasource);
    static int close_callback(void *datasource);

    OggVorbis_File vf;
    bool vf_valid;

public:
    vorbis_streamer(int _fd, int buffer_size, const struct lookahead &la);
    virtual ~vorbis_streamer();
    virtual enum read_status read(int samples);
    virtual void shutdown();
};

/* ------------------------------------------------------------------------- */

streamer::streamer(int _fd, int buffer_size)
{
    fd = _fd;
    buf = (char *)malloc(sizeof(sample_t) * buffer_size);
    rdptr = wrptr = buf;
    endptr = buf + sizeof(sample_t) * buffer_size;
    sample_offset = 0;
}

streamer::~streamer()
{
    free(buf);
}

void streamer::check_format()
{
    if(1 != num_channels && 2 != num_channels) {
        fputs("only mono or stereo streams are supported", stderr);
        exit(EXIT_FAILURE);
    }
    if(16 != bits_per_sample) {
        fputs("only 16 bits per sample streams are supported", stderr);
        exit(EXIT_FAILURE);
    }
}

void streamer::need_buffer_space(int frame_bytes)
{
    if(endptr - rdptr < frame_bytes)
    {
        char *low  = rdptr < wrptr ? rdptr : wrptr,
             *high = rdptr > wrptr ? rdptr : wrptr;

        memmove(buf, rdptr, high-rdptr);
        wrptr -= low - buf;
        rdptr -= low - buf;
    }
}

void streamer::make_profile(struct mfcc::profile *profile) const
{
    profile->sample_rate = get_sample_rate();
    profile->frame_length = (int)(config.frame_sec * profile->sample_rate);
    profile->frame_spacing = (int)(config.step_sec * profile->sample_rate);
    profile->num_channels = get_num_channels();

    profile->mel_filters = config.mel_filters;
    profile->mel_high_freq = config.mel_high_freq;
    profile->mel_power_threshold = config.mel_power_threshold;
}

/* -------------------------------------------------------------------------- */

wav_streamer::wav_streamer(int _fd, int buffer_size, const struct lookahead &la)
                    : streamer(_fd, buffer_size)
{
    if(la.match("RIFF"))
        parse_wav(la);
    else
        assume_raw(la);
    check_format();

    set_blocking(false);
}
wav_streamer::~wav_streamer() {
    shutdown();
}

void wav_streamer::parse_wav(const struct lookahead &la)
{
    struct wavhdr hdr;

    memcpy(&hdr, la.data, la.size);

    ssize_t remaining = sizeof(hdr) - la.size;
    char *hptr = (char *)&hdr + la.size;
    while(remaining > 0)
    {
        ssize_t rc = ::read(fd, hptr, remaining);
        if(-1 == rc) {
            perror("read");
            exit(EXIT_FAILURE);
        }
        if(0 == rc) {
            fputs("truncated wav file", stderr);
            exit(EXIT_FAILURE);
        }
        remaining -= rc;
        hptr += rc;
    }

    if(0 != memcmp(hdr.riff_format, "WAVE", 4) ||
       0 != memcmp(hdr.sub1_chunkid, "fmt ", 4) ||
       0 != memcmp(hdr.sub2_chunkid, "data", 4) ||
       16 != hdr.sub1_chunksize || 1 != hdr.audiofmt) {
        fputs("malformed or unrecognized wav header", stderr);
        exit(EXIT_FAILURE);
    }

    num_channels = hdr.num_channels;
    sample_rate = hdr.sample_rate;
    bits_per_sample = hdr.bits_per_sample;
}

void wav_streamer::assume_raw(const struct lookahead &la)
{
    num_channels = 1;
    sample_rate = 16000;
    bits_per_sample = 16;

    memcpy(wrptr, la.data, la.size);
    wrptr += la.size;
}

void wav_streamer::set_blocking(bool blocking)
{
    int new_flags = blocking
                       ? fd_flags & ~O_NONBLOCK
                       : fd_flags | O_NONBLOCK;
    if(fd_flags == new_flags)
        return;

    if(-1 == fcntl(STDIN_FILENO, F_SETFL, new_flags)) {
        perror("fcntl(F_SETFL)");
        exit(EXIT_FAILURE);
    }

    fd_flags = new_flags;
}

void wav_streamer::shutdown() {
    close(fd);
}

enum streamer::read_status wav_streamer::read(int samples)
{
    int frame_bytes = sizeof(sample_t) * samples * num_channels;

    need_buffer_space(frame_bytes);

    while(wrptr - rdptr < frame_bytes)
    {
        ssize_t rc = ::read(fd, wrptr, endptr - wrptr);

        if(0 == rc) {
            memset(wrptr, 0, endptr - wrptr);
            if(wrptr > rdptr)
                break;
            return streamer::READ_EOF;
        }

        if(-1 == rc)
        {
            if(EAGAIN == errno) {
                set_blocking(true);
                return streamer::READ_STALL;
            }
            perror("read");
            exit(EXIT_FAILURE);
        }

        wrptr += rc;
    }

    set_blocking(false);
    return streamer::READ_OK;
}

/* -------------------------------------------------------------------------- */

vorbis_streamer::vorbis_streamer(int _fd, int buffer_size, const struct lookahead &la)
                    : streamer(_fd, buffer_size)
{
    ov_callbacks callbacks = { };
    callbacks.read_func = &read_callback;
    callbacks.close_func = &close_callback;

    int rc = ov_open_callbacks(this, &vf, la.data, la.size, callbacks);
    if(0 != rc) {
        fprintf(stderr, "failed to open ogg file: ov_open_callbacks: %d\n", rc);
        exit(EXIT_FAILURE);
    }
    vf_valid = true;

    vorbis_info *info = ov_info(&vf, -1);
    num_channels = info->channels;
    sample_rate = info->rate;
    bits_per_sample = 16;

    check_format();
}
vorbis_streamer::~vorbis_streamer() {
    shutdown();
}

size_t vorbis_streamer::read_callback(void *ptr, size_t size, size_t nmemb, void *datasource)
{
    class vorbis_streamer *streamer = (class vorbis_streamer *)datasource;
    size_t toread = size * nmemb;

    ssize_t rc = ::read(streamer->fd, ptr, toread);
    if(-1 == rc)
        return 0;
    if(0 == rc) {
        errno = 0;
        return 0;
    }
    return toread / size;
}

int vorbis_streamer::close_callback(void *datasource) {
    close( ((class vorbis_streamer *)datasource)->fd );
    return 0;
}

void vorbis_streamer::shutdown() {
    if(!vf_valid)
        return;
    ov_clear(&vf);
    vf_valid = false;
}

enum streamer::read_status vorbis_streamer::read(int samples)
{
    int frame_bytes = sizeof(sample_t) * samples * num_channels;
    
    need_buffer_space(frame_bytes);

    while(wrptr - rdptr < frame_bytes)
    {
        int bitstream_id;

        long rc = ov_read(&vf, wrptr, endptr - wrptr, 0,2,1, &bitstream_id);
        if(-1 == rc) {
            fputs("failed to read data from ogg file\n", stderr);
            exit(EXIT_FAILURE);
        }
        if(0 == rc) {
            memset(wrptr, 0, endptr - wrptr);
            if(wrptr > rdptr)
                break;
            return streamer::READ_EOF;
        }

        wrptr += rc;
    }

    return streamer::READ_OK;
}

/* -------------------------------------------------------------------------- */

class outstream
{
    FILE *fp;
    bool fft;

    inline FILE *get_fp() {
        if(NULL == fp)
            fp = stdout;
        return fp;
    }

    enum packets {
        PACKET_PROFILE = 1,
        PACKET_GROUP_HDR = 2,
        PACKET_FRAME = 3
    };

    inline void out_buf(const void *data, size_t len) {
        fwrite(data, len, 1, get_fp());
    }
    inline void out_int(uint32_t x) {
        out_buf(&x, sizeof(x));
    }
    inline void out_short(int16_t x) {
        out_buf(&x, sizeof(x));
    }
    inline void out_byte(int8_t x) {
        out_buf(&x, sizeof(x));
    }

public:
    outstream();
    outstream(FILE *_fp);

    inline void flush() {
        fflush(get_fp());
    }

    void write_profile(const mfcc &mfcc, bool _fft);
    void write_group_hdr(const char *filename, const char *label, int sample_offset);
    void write_frame(const mfcc &mfcc);
};

static outstream out;

outstream::outstream() {
    fp = NULL;
    fft = true;
}
outstream::outstream(FILE *_fp) {
    fp = _fp;
    fft = true;
}
void outstream::write_profile(const mfcc &mfcc, bool _fft)
{
    fft = _fft;

    out_byte(PACKET_PROFILE);

    out_byte(mfcc.p.mel_filters);
    out_short(fft ? mfcc.fft_length : 0);
    out_short(mfcc.p.frame_length);
    out_short(mfcc.p.frame_spacing);
    out_short(mfcc.p.sample_rate);

    out_buf(mfcc.mel_freqs, sizeof(float)*(mfcc.p.mel_filters+2));
    if(fft) out_buf(mfcc.fft_freqs, sizeof(float)*mfcc.fft_length);
}
void outstream::write_group_hdr(const char *filename, const char *label, int sample_offset)
{
    int filename_len = strlen(filename),
        label_len = strlen(label);

    out_byte(PACKET_GROUP_HDR);

    out_byte(filename_len);
    out_byte(label_len);
    out_int(sample_offset);

    out_buf(filename, filename_len);
    out_buf(label, label_len);
}
void outstream::write_frame(const mfcc &mfcc)
{
    out_byte(PACKET_FRAME);

    out_buf(mfcc.mel_power,  sizeof(float)*mfcc.p.mel_filters);
    if(fft) out_buf(mfcc.fft_power,  sizeof(float)*mfcc.fft_length);
    out_buf(mfcc.dct_coeffs, sizeof(float)*mfcc.p.mel_filters);
    out_buf(mfcc.wvl_coeffs, sizeof(float)*mfcc.p.mel_filters);
}

/* -------------------------------------------------------------------------- */

static void get_fd_filename(int fd, char *buf, int bufsize)
{
    char procfspath[32];
    sprintf(procfspath, "/proc/self/fd/%d", fd);

    ssize_t len = readlink(procfspath, buf, bufsize);
    if(-1 == len) {
        perror("failed to resolve fd to pathname: readlink");
        exit(EXIT_FAILURE);
    }
    if(len >= bufsize) {
        fputs("real path of fd too long", stderr);
        exit(EXIT_FAILURE);
    }
    buf[len] = '\0';
}

class label_source
{
protected:
    inline label_source() { }

public:
    struct source {
        class streamer *streamer;
        char name[256];
    };

    struct label {
        char name[32];
        long long start;
        long long end;
    };

    virtual ~label_source();

    virtual bool next_source(struct source *src) = 0;
    virtual bool next_label(struct label *lbl) = 0;
};

label_source::~label_source() {
}

class simple_label_source : public label_source
{
    int fd;
    struct lookahead la;
    char name_hint[256];
    bool got_name_hint;
    bool has_source, has_label;

public:
    simple_label_source(int _fd, const struct lookahead &_la, const char *_name_hint);
    virtual ~simple_label_source();

    virtual bool next_source(struct source *src);
    virtual bool next_label(struct label *lbl);
};

class mlf_label_source : public label_source
{
    FILE *fp;
    char buffer[256], basedir[256], filename[256];
    bool has_line;

    bool get_line();

public:
    mlf_label_source(int _fd, const struct lookahead &_la);
    virtual ~mlf_label_source();

    virtual bool next_source(struct source *src);
    virtual bool next_label(struct label *lbl);
};

class args_label_source : public label_source
{
    int argc;
    char** argv;
    int argidx;

    class label_source *child;

public:
    args_label_source(int _argc, char **_argv);
    virtual ~args_label_source();

    virtual bool next_source(struct source *src);
    virtual bool next_label(struct label *lbl);
};


simple_label_source::simple_label_source(int _fd, const struct lookahead &_la, const char *_name_hint)
        : fd(_fd), la(_la), has_source(true), has_label(true)
{
    got_name_hint = (NULL != _name_hint);
    if(got_name_hint)
        strcpy(name_hint, _name_hint);
}
simple_label_source::~simple_label_source() {
    if(-1 != fd)
        close(fd);
}

bool simple_label_source::next_source(struct source *src)
{
    if(!has_source)
        return false;

    if(got_name_hint)
        strcpy(src->name, name_hint);
    else
        get_fd_filename(fd, src->name, sizeof(src->name));

    if(la.size >= 4 && 0 == memcmp(la.data, "OggS", 4))
        src->streamer = new vorbis_streamer(fd, config.streamer_buffer, la);
    else
        src->streamer = new wav_streamer(fd, config.streamer_buffer, la);

    fd = -1;
    has_source = false;
    return true;
}

bool simple_label_source::next_label(struct label *lbl)
{
    if(has_source || !has_label)
        return false;

    strcpy(lbl->name, "?");
    lbl->start = 0ll;
    lbl->end = 0x7fffffffffffffffll;

    has_label = false;
    return true;
}


mlf_label_source::mlf_label_source(int _fd, const struct lookahead &_la)
{
    get_fd_filename(_fd, basedir, sizeof(basedir));
    if(basedir[0] != '/')
        basedir[0] = '\0';
    else
        strrchr(basedir, '/')[1] = '\0';

    fp = fdopen(_fd, "r");
    if(NULL == fp) {
        perror("fdopen");
        exit(EXIT_FAILURE);
    }
}

mlf_label_source::~mlf_label_source() {
    fclose(fp);
}

bool mlf_label_source::get_line()
{
    if(has_line)
        return true;
    do {
        if(NULL == fgets(buffer, sizeof(buffer), fp))
        {
            if(feof(fp))
                return false;
            perror("fgets");
            exit(EXIT_FAILURE);
        }
    } while(buffer[0] == '\n' || buffer[0] == '\r');
    has_line = true;
    return true;

}

bool mlf_label_source::next_source(struct source *src)
{
    while(get_line())
    {
        if('.' == buffer[0]) {
            has_line = false;
            continue;
        }

        if('"' == buffer[0])
        {
            char *f = buffer+3,
                 *e = strchr(f, '.');
            strcpy(e, ".ogg");
            strcpy(src->name, basedir);
            strcat(src->name, f);

            int fd = open(src->name, O_RDONLY);
            if(-1 == fd) {
                perror("open");
                exit(EXIT_FAILURE);
            }

            has_line = false;

            simple_label_source s(fd, lookahead(fd), f);
            return s.next_source(src);
        }

        has_line = false;
    }

    return false;
}

bool mlf_label_source::next_label(struct label *lbl)
{
    while(get_line())
    {
        if('.' == buffer[0]) {
            has_line = false;
            continue;
        }

        if('"' == buffer[0])
            break;

        sscanf(buffer, "%lld %lld %s", &lbl->start, &lbl->end, lbl->name);
        has_line = false;
        return true;
    }

    return false;
}

args_label_source::args_label_source(int _argc, char **_argv)
        : argc(_argc), argv(_argv), argidx(0), child(NULL) { }
args_label_source::~args_label_source() {
    if(NULL != child)
        delete child;
}

bool args_label_source::next_source(struct source *src)
{
    if(NULL != child) {
        if(child->next_source(src))
            return true;
        delete child;
        child = NULL;
    }
    
    if(argidx >= argc)
        return false;

    int fd;
    const char *name_hint;
    
    if(0 == strcmp("-", argv[argidx])) {
        fd = STDIN_FILENO;
        name_hint = NULL;
    } else {
        fd = open(argv[argidx], O_RDONLY);
        if(-1 == fd) {
            perror("open");
            exit(EXIT_FAILURE);
        }
        name_hint = argv[argidx];
    }
    
    struct lookahead la(fd);

    if(la.match("#!MLF!#"))
        child = new mlf_label_source(fd, la);
    else
        child = new simple_label_source(fd, la, name_hint);

    argidx += 1;

    return next_source(src);
}

bool args_label_source::next_label(struct label *lbl) {
    return NULL != child && child->next_label(lbl);
}

/* -------------------------------------------------------------------------- */

static void convert(int argc, char *argv[])
{
    argc -= 1; argv += 1;

    bool write_fft = true;
    if(argc >= 1 && 0 == strcmp(argv[0], "--no-fft")) {
        write_fft = false;
        argc -= 1; argv += 1;
    }

    args_label_source lblsrc(argc, argv);

    struct label_source::source src;
    struct label_source::label lbl;
    struct mfcc::profile profile;
    class mfcc *mfcc = NULL;

    while(lblsrc.next_source(&src))
    {
        fprintf(stderr, "processing file %s\n", src.name);

        struct mfcc::profile new_profile;
        src.streamer->make_profile(&new_profile);
        
        if(NULL == mfcc || new_profile != profile)
        {
            profile = new_profile;

            if(NULL != mfcc)
                delete mfcc;
            mfcc = new class mfcc(profile);

            out.write_profile(*mfcc, write_fft);

            char buf[512], *ptr = buf;
            ptr += sprintf(ptr,
                    "profile: sample rate %d Hz; frame length: %d samples, frame spacing: %d samples\n"
                    "%d mel filters:",
                    profile.sample_rate, profile.frame_length, profile.frame_spacing, profile.mel_filters);
            for(int i=0; i<profile.mel_filters+2; i++)
                ptr += sprintf(ptr, " %.1f Hz", mfcc->mel_freqs[i]);
            ptr += sprintf(ptr, "\n");
            fputs(buf, stderr);
        }

        float frame_duration = (float)profile.frame_length / profile.sample_rate;

        while(lblsrc.next_label(&lbl))
        {
            if(lbl.start == lbl.end)
                continue;

            bool need_header = true;
            float lbl_start = lbl.start * 1e-7f,
                  lbl_end = lbl.end * 1e-7f;

            for(;;)
            {
                enum streamer::read_status rc = src.streamer->read(profile.frame_length);
                if(rc == streamer::READ_EOF)
                    break;
                if(rc == streamer::READ_STALL) {
                    out.flush();
                    continue;
                }

                int sample_offset = src.streamer->get_sample_offset();
                float time_offset = (float)sample_offset / profile.sample_rate;
                
                if(time_offset + frame_duration <= lbl_start) {
                    src.streamer->advance(profile.frame_spacing);
                    continue;
                }

                if(time_offset >= lbl_end)
                    break;

                if(need_header) {
                    out.write_group_hdr(src.name, lbl.name, sample_offset);
                    need_header = false;
                }

                mfcc->process_frame(src.streamer->get_samples());
                src.streamer->advance(profile.frame_spacing);
                out.write_frame(*mfcc);
            }

            if(need_header)
                fprintf(stderr, "warning: label %s (range %lld -- %lld) in file %s did not hit any frame\n", 
                lbl.name, lbl.start, lbl.end, src.name);
        }

        delete src.streamer;
    }
}

/* -------------------------------------------------------------------------- */

int main(int argc, char *argv[])
{
    convert(argc, argv);
    return EXIT_SUCCESS;
}

