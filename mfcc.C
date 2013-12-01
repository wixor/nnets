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
    int dct_length;
} config = {
    .streamer_buffer = 8192,
    .frame_sec = 0.025f,
    .step_sec = 0.010f,
    .mel_filters = 26,
    .mel_high_freq = 8000.f,
    .mel_power_threshold = -70.f,
    .dct_length = 13,
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
        int dct_length;
        
        bool operator==(const struct profile &p) const
        {
            return sample_rate == p.sample_rate &&
                   frame_length == p.frame_length &&
                   frame_spacing == p.frame_spacing &&
                   num_channels == p.num_channels &&
                   mel_filters == p.mel_filters &&
                   mel_high_freq == p.mel_high_freq &&
                   mel_power_threshold == p.mel_power_threshold &&
                   dct_length == p.dct_length;
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

    int dct_length;
    double *dct_in, *dct_out;
    fftw_plan dct_plan;

    float *window;
    float *fft_freqs, *fft_power;
    float *mel_freqs, *mel_power;
    float *dct_coeffs;


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
   
    dct_length = p.mel_filters / 2; 
    dct_coeffs = (float *)malloc(sizeof(float) * dct_length);

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
    for(int i=0; i<dct_length; i++)
        dct_coeffs[i] = (float)(dct_out[i] / (2*p.mel_filters));
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

public:
    enum read_status {
        READ_OK, READ_STALL, READ_EOF
    };

    static class streamer *init(int fd, int buffer_size);

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

    void parse_wav(char *lookahead, size_t lookahead_size);
    void assume_raw(char *lookahead, size_t lookahead_size);
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
    wav_streamer(int _fd, int buffer_size, char *lookahead, size_t lookahead_size);
    virtual ~wav_streamer();
    virtual enum read_status read(int samples);
    virtual void shutdown();
};

class vorbis_streamer : public streamer
{
protected:
    static size_t read_callback(void *ptr, size_t size, size_t nmemb, void *datasource);

    OggVorbis_File vf;
    bool vf_valid;

public:
    vorbis_streamer(int _fd, int buffer_size, char *lookahead, size_t lookahead_size);
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

class streamer* streamer::init(int fd, int buffer_size)
{
    char lookahead[4];
    int rc = ::read(fd, lookahead, sizeof(lookahead));
    if(-1 == rc) {
        perror("failed to recognize file format: read");
        exit(EXIT_FAILURE);
    }
    if(sizeof(lookahead) != rc) {
        fputs("file too short", stderr);
        exit(EXIT_FAILURE);
    }

    class streamer *ret;
    if(0 == memcmp(lookahead, "OggS", 4))
        ret = new vorbis_streamer(fd, buffer_size, lookahead, sizeof(lookahead));
    else
        ret = new wav_streamer(fd, buffer_size, lookahead, sizeof(lookahead));

    if(1 != ret->num_channels && 2 != ret->num_channels) {
        fputs("only mono or stereo streams are supported", stderr);
        exit(EXIT_FAILURE);
    }
    if(16 != ret->bits_per_sample) {
        fputs("only 16 bits per sample streams are supported", stderr);
        exit(EXIT_FAILURE);
    }

    return ret;
}

void streamer::need_buffer_space(int frame_bytes)
{
    if(endptr - rdptr < frame_bytes)
    {
        memmove(buf, rdptr, wrptr - rdptr);
        wrptr = buf + (wrptr - rdptr);
        rdptr = buf;
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
    profile->dct_length = config.dct_length;
}

/* -------------------------------------------------------------------------- */

wav_streamer::wav_streamer(int _fd, int buffer_size, char *lookahead, size_t lookahead_size)
                    : streamer(_fd, buffer_size)
{
    if(0 == memcmp(lookahead, "RIFF", 4))
        parse_wav(lookahead, lookahead_size);
    else
        assume_raw(lookahead, lookahead_size);

    set_blocking(false);
}
wav_streamer::~wav_streamer() {
    shutdown();
}

void wav_streamer::parse_wav(char *lookahead, size_t lookahead_size)
{
    struct wavhdr hdr;

    memcpy(&hdr, lookahead, lookahead_size);

    ssize_t remaining = sizeof(hdr) - lookahead_size;
    while(remaining > 0)
    {
        ssize_t rc = ::read(fd, (char *)&hdr + lookahead_size, remaining);
        if(-1 == rc) {
            perror("read");
            exit(EXIT_FAILURE);
        }
        if(0 == rc) {
            fputs("truncated wav file", stderr);
            exit(EXIT_FAILURE);
        }
        remaining -= rc;
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

void wav_streamer::assume_raw(char *lookahead, size_t lookahead_size)
{
    num_channels = 1;
    sample_rate = 16000;
    bits_per_sample = 16;

    memcpy(wrptr, lookahead, lookahead_size);
    wrptr += lookahead_size;
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
    set_blocking(true);
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

vorbis_streamer::vorbis_streamer(int _fd, int buffer_size, char *lookahead, size_t lookahead_size)
                    : streamer(_fd, buffer_size)
{
    ov_callbacks callbacks = { .read_func = &read_callback };

    int rc = ov_open_callbacks(this, &vf, lookahead, lookahead_size, callbacks);
    if(0 != rc) {
        fprintf(stderr, "failed to open ogg file: ov_open_callbacks: %d\n", rc);
        exit(EXIT_FAILURE);
    }
    vf_valid = true;

    vorbis_info *info = ov_info(&vf, -1);
    num_channels = info->channels;
    sample_rate = info->rate;
    bits_per_sample = 16;

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

    void write_profile(const mfcc &mfcc);
    void write_group_hdr(const char *filename, const char *label, int sample_offset);
    void write_frame(const mfcc &mfcc);
};

static outstream out;

outstream::outstream() {
    fp = NULL;
}
outstream::outstream(FILE *_fp) {
    fp = _fp;
}
void outstream::write_profile(const mfcc &mfcc)
{
    out_byte(PACKET_PROFILE);

    out_short(mfcc.p.frame_length);
    out_short(mfcc.p.frame_spacing);
    out_short(mfcc.p.sample_rate);
    out_byte(mfcc.p.mel_filters);
    out_byte(mfcc.dct_length);
    out_short(mfcc.fft_length);

    out_buf(mfcc.mel_freqs, sizeof(float)*mfcc.p.mel_filters);
    out_buf(mfcc.fft_freqs, sizeof(float)*mfcc.fft_length);
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
    out_buf(mfcc.fft_power,  sizeof(float)*mfcc.fft_length);
    out_buf(mfcc.dct_coeffs, sizeof(float)*mfcc.dct_length);
}

/* -------------------------------------------------------------------------- */

static int sub_pipe(int argc, char *argv[])
{
    class streamer *streamer = streamer::init(STDIN_FILENO, config.streamer_buffer);

    struct mfcc::profile profile;
    streamer->make_profile(&profile);

    class mfcc mfcc(profile);

    {
        char buf[512], *ptr = buf;

        ptr += sprintf(ptr,
                "sample rate %d Hz; frame length: %d samples, frame spacing: %d samples\n"
                "%d mel filters:",
                profile.sample_rate, profile.frame_length, profile.frame_spacing, profile.mel_filters);

        for(int i=0; i<profile.mel_filters+2; i++)
            ptr += sprintf(ptr, " %.1f Hz", mfcc.mel_freqs[i]);
        ptr += sprintf(ptr, "\n%d dct coefficients\n", mfcc.dct_length);

        fputs(buf, stderr);
    }

    out.write_profile(mfcc);
    out.write_group_hdr("pipe", "?", 0);

    int frame_count = 0;
    for(;;)
    {
        enum streamer::read_status rc = streamer->read(profile.frame_length);
        if(rc == streamer::READ_EOF)
            break;
        if(rc == streamer::READ_STALL)
            out.flush();
        else {
            mfcc.process_frame(streamer->get_samples());
            streamer->advance(profile.frame_spacing);

            out.write_frame(mfcc);
            frame_count += 1;
        }
    }

    out.flush();

    fprintf(stderr, "processed %d frames\n", frame_count);

    delete streamer;

    return 0;
}

/* -------------------------------------------------------------------------- */

class mlf_parser
{
public:
    enum line_type {
        FILE_START, LABEL, FILE_END, NO_LINE
    };

private:
    FILE *fp;
    char buffer[128];

    enum line_type type;
    const char *filename;
    long long label_start;
    long long label_end;
    const char *label_value;

    inline void read_line() {
        buffer[0] = '\0';
        fgets(buffer, sizeof(buffer), fp);
    }

public:
    mlf_parser(const char *mlffile);
    ~mlf_parser();

    bool next_line();
    inline enum line_type get_line_type() const { return type; }
    inline const char* get_filename() const { return filename; }
    inline long long get_label_start() const { return label_start; }
    inline long long get_label_end() const { return label_end; }
    inline const char *get_label_value() const { return label_value; }
};

mlf_parser::mlf_parser(const char *mlffile)
{
    fp = fopen(mlffile, "r");
    if(NULL == fp) {
        perror("failed to open mlf file: fopen");
        exit(EXIT_FAILURE);
    }

    read_line();

    if(0 != memcmp("#!MLF!#", buffer, 7)) {
        fputs("mlf signature missing", stderr);
        exit(EXIT_FAILURE);
    }
}
mlf_parser::~mlf_parser() {
    fclose(fp);
}

bool mlf_parser::next_line()
{
    do 
        read_line();
    while('\n' == buffer[0] || '\r' == buffer[0]);

    if('\0' == buffer[0]) {
        type = mlf_parser::NO_LINE;
        filename = label_value = NULL;
        label_start = label_end = 0LL;
        return false;
    }

    if('.' == buffer[0]) {
        type = mlf_parser::FILE_END;
        filename = label_value = NULL;
        label_start = label_end = 0LL;
        return true;
    }
    if('"' == buffer[0]) {
        type = mlf_parser::FILE_START;
        
        filename = buffer+3;
        *strchrnul(buffer, '.') = '\0';
        
        label_value = NULL;
        label_start = label_end = 0LL;
        return true;
    }
    {
        type = mlf_parser::LABEL;
        filename = NULL;

        const char *delim = " \n\t\r";
        char *saveptr, *p;
        p = strtok_r(buffer, delim, &saveptr);
        label_start = strtoll(p, NULL, 10);

        p = strtok_r(NULL, delim, &saveptr);
        label_end = strtoll(p, NULL, 10);

        p = strtok_r(NULL, delim, &saveptr);
        label_value = p;
        return true;
    }
}

class sub_corpora
{
    char basedir[128];
    char path[128];

    int fd;
    class streamer *streamer;
    class mfcc *mfcc;
    struct mfcc::profile profile;

    sub_corpora();
    ~sub_corpora();

    void drop_streamer();
    void drop_mfcc();

    int run(int argc, char *argv[]);

    void open_file(const char *filename);
    void process_slice(long long label_start, long long label_end, const char *label_value);

public:
    static inline int main(int argc, char *argv[]) {
        return sub_corpora().run(argc, argv);
    }
};

sub_corpora::sub_corpora() {
    fd = -1;
    streamer = NULL;
    mfcc = NULL;
}
sub_corpora::~sub_corpora() {
    drop_streamer();
    drop_mfcc();
}

void sub_corpora::drop_streamer() {
    if(-1 != fd) {
        close(fd);
        fd = -1;
    }
    if(NULL != streamer) {
        delete streamer;
        streamer = NULL;
    }
}
void sub_corpora::drop_mfcc() {
    if(NULL != mfcc) {
        delete mfcc;
        mfcc = NULL;
    }
}

int sub_corpora::run(int argc, char *argv[])
{
    if(argc != 3) {
        fputs("USAGE: mfcc corpora [mlf file]\n", stderr);
        exit(EXIT_FAILURE);
    }

    {
        strcpy(basedir, argv[2]);
        char *slash = strrchr(basedir, '/');
        if(NULL != slash)
            slash[1] = '\0';
        else
            basedir[0] = '\0';

        fprintf(stderr, "opening MLF file %s; base directory: %s\n", argv[2], basedir);
    }

    mlf_parser mlf(argv[2]);

    while(mlf.next_line())
        switch(mlf.get_line_type()) {
            case mlf_parser::FILE_START:
                open_file(mlf.get_filename());
                break;
            case mlf_parser::FILE_END:
                drop_streamer();
                break;
             case mlf_parser::LABEL:
                process_slice(mlf.get_label_start(), mlf.get_label_end(), mlf.get_label_value());
                break;
            default:
                break;
        }

    return 0;
}

void sub_corpora::open_file(const char *filename)
{
    drop_streamer();

    sprintf(path, "%s%s.ogg", basedir, filename);
    fprintf(stderr, "opening file %s\n", path);

    fd = open(path, O_RDONLY);
    if(-1 == fd) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    streamer = streamer::init(fd, config.streamer_buffer);

    struct mfcc::profile new_profile;
    streamer->make_profile(&new_profile);

    if(NULL == mfcc || new_profile != profile)
    {
        fputs("creating new mfcc profile...\n", stderr);
        profile = new_profile;

        drop_mfcc();
        mfcc = new class mfcc(profile);

        out.write_profile(*mfcc);
    }
}

void sub_corpora::process_slice(long long label_start, long long label_end, const char *label_value)
{
    if(label_start == label_end)
        return;

    float lbl_start = label_start * 1e-7f,
          lbl_end = label_end * 1e-7f;

    float frame_duration = (float)profile.frame_length / profile.sample_rate;

    bool need_header = true;

    for(;;)
    {
        enum streamer::read_status rc = streamer->read(profile.frame_length);
        if(rc == streamer::READ_EOF)
            break;

        int sample_offset = streamer->get_sample_offset();
        float time_offset = (float)sample_offset / profile.sample_rate;
        
        if(time_offset + frame_duration <= lbl_start) {
            streamer->advance(profile.frame_spacing);
            continue;
        }

        if(time_offset >= lbl_end)
            break;

        if(need_header) {
            out.write_group_hdr(path, label_value, sample_offset);
            need_header = false;
        }

        mfcc->process_frame(streamer->get_samples());
        streamer->advance(profile.frame_spacing);
        out.write_frame(*mfcc);
    }

    if(need_header)
        fprintf(stderr, "warning: label %s (range %lld -- %lld) in file %s did not hit any frame\n", 
                label_value, label_start, label_end, path);
}

/* -------------------------------------------------------------------------- */

int main(int argc, char *argv[])
{
    if(argc < 2) {
        fputs("USAGE: mfcc [subprogram] [options...]\n", stderr);
        exit(EXIT_FAILURE);
    }

    if(strcmp(argv[1], "pipe") == 0)
        return sub_pipe(argc, argv);
    if(strcmp(argv[1], "corpora") == 0)
        return sub_corpora::main(argc, argv);

    fputs("unknown subprogram; available: pipe, corpora\n", stderr);
    exit(EXIT_FAILURE);
}

