#define _CRT_SECURE_NO_WARNINGS

#include <vector>
#include <random>
#include <array>
#include <string.h>
#include <direct.h>
#include <stdarg.h>
#include <filesystem>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static const float c_pi = 3.14159265359f;
static const float c_goldenRatioConjugate = 0.61803398875f;

static const size_t c_numSamples = 256;
static const size_t c_numTests = 10;

#define DETERMINISTIC() true   // if true, all randomization will be the same every time.

// ====================================================================
//                          INTERNAL
// ====================================================================

inline float sRGBToLinear(float value)
{
    if (value < 0.04045f)
        return value / 12.92f;
    else
        return std::powf(((value + 0.055f) / 1.055f), 2.4f);
}

inline float LinearTosRGB(float value)
{
    if (value < 0.0031308f)
        return value * 12.92f;
    else
        return std::powf(value, 1.0f / 2.4f) * 1.055f - 0.055f;
}

float Fract(float f)
{
    return f - std::floor(f);
}

float Lerp(float a, float b, float t)
{
    return a * (1.0f - t) + b * t;
}

float Clamp(float v, float min, float max)
{
    if (v <= min)
        return min;
    else if (v >= max)
        return max;
    else
        return v;
}

using Vec2 = std::array<float, 2>;

template <size_t N>
std::array<float, N> Fract(const std::array<float, N>& v)
{
    std::array<float, N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = Fract(v[i]);
    return ret;
}

Vec2 operator + (const Vec2& A, const Vec2& B)
{
    return Vec2{ A[0] + B[0], A[1] + B[1] };
}

std::mt19937 GetRNG(int seed)
{
#if DETERMINISTIC()
    std::mt19937 mt(seed);
#else
    std::random_device rd("/dev/random");
    std::mt19937 mt(rd());
#endif
    return mt;
}

typedef std::vector<std::vector<std::string>> CSV;

void SetCSV(CSV& csv, size_t x, size_t y, const char* format, ...)
{
    if (csv.size() <= y)
        csv.resize(y + 1);

    if (csv[y].size() <= x)
        csv[y].resize(x + 1);

    char buffer[4096];
    va_list args;
    va_start(args, format);
    vsprintf_s(buffer, format, args);
    csv[y][x] = buffer;
    va_end(args);
}

size_t GetCSVCols(const CSV& csv)
{
    if (csv.size() == 0)
        return 0;
    return csv[0].size();
}

void WriteCSV(const CSV& csv, const char* fileNameFormat, ...)
{
    char buffer[4096];
    va_list args;
    va_start(args, fileNameFormat);
    vsprintf_s(buffer, fileNameFormat, args);
    va_end(args);

    FILE* file = nullptr;
    fopen_s(&file, buffer, "w+t");

    for (const std::vector<std::string>& row : csv)
    {
        bool first = true;
        for (const std::string& item : row)
        {
            fprintf(file, "%s\"%s\"", first ? "" : ",", item.c_str());
            first = false;
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

// ====================================================================
//                     SAMPLING FUNCTIONS
// ====================================================================

using SamplesFn = void (*)(std::vector<Vec2>& points, size_t count, int seed);

struct Samples
{
    const char* name = nullptr;
    SamplesFn function;
    bool deterministic = true;
    bool progressive = true;
};

void Samples_WhiteNoise(std::vector<Vec2>& points, size_t count, int seed)
{
    std::mt19937 rng = GetRNG(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    points.resize(count);
    for (Vec2& v : points)
    {
        v[0] = dist(rng);
        v[1] = dist(rng);
    }
}

float ToroidalDistance(const Vec2& A, const Vec2& B)
{
    Vec2 V;
    V[0] = A[0] - B[0];
    V[1] = A[1] - B[1];

    float distSq = 0.0f;

    for (float dist1D : V)
    {
        dist1D = abs(dist1D);
        if (dist1D > 0.5f)
            dist1D = 1.0f - dist1D;
        distSq += dist1D * dist1D;
    }

    return std::sqrt(distSq);
}

void Samples_BlueNoise(std::vector<Vec2>& points, size_t count, int seed)
{
    std::mt19937 rng = GetRNG(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    points.resize(count);
    for (size_t sampleIndex = 0; sampleIndex < count; ++sampleIndex)
    {
        // generate candidates and keep the one that is farthest away from it's closest neighbor
        Vec2 bestCandidate = {};
        float bestCandidateScore = -FLT_MAX;
        const size_t numCandidates = sampleIndex + 1;
        for (size_t candidateIndex = 0; candidateIndex < numCandidates; ++candidateIndex)
        {
            Vec2 candidate;
            for (float& f : candidate)
                f = dist(rng);

            float minDist = FLT_MAX;
            for (size_t pointIndex = 0; pointIndex < sampleIndex; ++pointIndex)
            {
                float dist = ToroidalDistance(points[pointIndex], candidate);
                minDist = std::min(minDist, dist);
            }

            if (minDist > bestCandidateScore)
            {
                bestCandidate = candidate;
                bestCandidateScore = minDist;
            }
        }
        points[sampleIndex] = bestCandidate;
    }
}

void Samples_R2(std::vector<Vec2>& points, size_t count, int seed)
{
    // generalized golden ratio, from:
    // http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    const float c_R2_g = 1.32471795724474602596f;
    const float c_R2_a1 = 1.0f / c_R2_g;
    const float c_R2_a2 = 1.0f / (c_R2_g * c_R2_g);

    points.resize(count);
    float lastValueX = 0.5f;
    float lastValueY = 0.5f;
    for (size_t i = 0; i < count; ++i)
    {
        lastValueX = Fract(lastValueX + c_R2_a1);
        lastValueY = Fract(lastValueY + c_R2_a2);
        points[i][0] = lastValueX;
        points[i][1] = lastValueY;
    }
}

size_t Ruler(size_t n)
{
    size_t ret = 0;
    while (n != 0 && (n & 1) == 0)
    {
        n /= 2;
        ++ret;
    }
    return ret;
}

void Samples_Sobol(std::vector<Vec2>& points, size_t count, int seed)
{
    // x axis
    points.resize(count);
    size_t sampleInt = 0;
    for (size_t i = 0; i < count; ++i)
    {
        size_t ruler = Ruler(i + 1);
        size_t direction = size_t(size_t(1) << size_t(31 - ruler));
        sampleInt = sampleInt ^ direction;
        points[i][0] = float(sampleInt) / std::pow(2.0f, 32.0f);
    }

    // y axis
    // Code adapted from http://web.maths.unsw.edu.au/~fkuo/sobol/
    // uses numbers: new-joe-kuo-6.21201

    // Direction numbers
    std::vector<size_t> V;
    V.resize((size_t)ceil(log((double)count + 1) / log(2.0)));  //+1 because we are skipping index 0
    V[0] = size_t(1) << size_t(31);
    for (size_t i = 1; i < V.size(); ++i)
        V[i] = V[i - 1] ^ (V[i - 1] >> 1);

    // Samples
    sampleInt = 0;
    for (size_t i = 0; i < count; ++i) {
        size_t ruler = Ruler(i + 1);
        sampleInt = sampleInt ^ V[ruler];
        points[i][1] = float(sampleInt) / std::pow(2.0f, 32.0f);
    }
}

void VanDerCorput(std::vector<Vec2>& values, size_t base, int axis, bool skipZero)
{
    // figure out how many bits we are working in.
    size_t numValues = values.size();
    size_t value = 1;
    size_t numBits = 0;
    while (value < numValues)
    {
        value *= 2;
        ++numBits;
    }
    size_t bitsMask = numBits > 0 ? (1 << numBits) - 1 : 0;

    for (size_t i = 0; i < values.size(); ++i)
    {
        values[i][axis] = 0.0f;
        float denominator = float(base);
        size_t n = i + (skipZero ? 1 : 0);
        n &= bitsMask;
        while (n > 0)
        {
            size_t multiplier = n % base;
            values[i][axis] += float(multiplier) / denominator;
            n = n / base;
            denominator *= base;
        }
    }
}

void Samples_Halton_2_3(std::vector<Vec2>& points, size_t count, int seed)
{
    points.resize(count);
    VanDerCorput(points, 2, 0, true);
    VanDerCorput(points, 3, 1, true);
}

Samples g_samples[] =
{
    {"White Noise", Samples_WhiteNoise, false, true},
    {"Blue Noise", Samples_BlueNoise, false, true},
    {"Halton(2,3)", Samples_Halton_2_3, true, true},
    {"Sobol", Samples_Sobol, true, true},
    {"R2", Samples_R2, true, true},
};

// ====================================================================

// actual value of the function integrated over the square
static const float c_GaussianActualValue = c_pi / 4.0f * (float)erf(1.0) * (float)erf(1.0);

float Gaussian(const Vec2& _samplePos, const Vec2& offset)
{
    Vec2 samplePos = Fract(_samplePos + offset);
    return expf(-(samplePos[0] * samplePos[0]) - (samplePos[1] * samplePos[1]));
}

static const float c_TriangleActualValue = 0.5f;

float Triangle(const Vec2& _samplePos, const Vec2& offset)
{
    Vec2 samplePos = Fract(_samplePos + offset);
    return (samplePos[0] < samplePos[1]) ? 0.0f : 1.0f;
}

static const float c_StepActualValue = c_goldenRatioConjugate;

float Step(const Vec2& _samplePos, const Vec2& offset)
{
    Vec2 samplePos = Fract(_samplePos + offset);
    return (samplePos[0] < c_goldenRatioConjugate) ? 1.0f : 0.0f;
}


template <typename LAMBDA>
void DoTest(const LAMBDA& lambda, const float c_actualValue, const Samples& samples, std::vector<float>& absError, std::vector<float>& sqAbsError, size_t testIndex)
{
    // make sure the output arrays are the right size and initialized
    absError.resize(c_numSamples, 0.0f);
    sqAbsError.resize(c_numSamples, 0.0f);

    // generate sample points
    std::vector<Vec2> samplePoints;
    samples.function(samplePoints, c_numSamples, (int)testIndex);

    // generate a random offset for deterministic samples
    Vec2 offset{ 0.0f, 0.0f };
    if (samples.deterministic)
    {
        std::mt19937 rng = GetRNG(int(testIndex));
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        offset = Vec2{ dist(rng), dist(rng) };
    }

    // do sampling
    float value = 0.0f;
    for (size_t index = 0; index < c_numSamples; ++index)
    {
        // monte carlo integrate into value using these samples
        float sampleValue = lambda(samplePoints[index], offset);
        value = Lerp(value, sampleValue, 1.0f / float(index + 1));

        // calculate the abs error
        float abserr = std::abs(value - c_actualValue);

        // calculate the average abs error and squared abs error for this sample count
        absError[index] = Lerp(absError[index], abserr, 1.0f / float(testIndex + 1));
        sqAbsError[index] = Lerp(sqAbsError[index], abserr * abserr, 1.0f / float(testIndex + 1));
    }
}

template <typename LAMBDA>
void MakeIntegrandImage(const LAMBDA& lambda, const char* outputFileName)
{
    static const int c_size = 64;
    std::vector<unsigned char> pixels(c_size * c_size);
    unsigned char* pixel = pixels.data();
    for (int y = 0; y < c_size; ++y)
    {
        float py = float(y) / float(c_size);
        for (int x = 0; x < c_size; ++x)
        {
            float px = float(x) / float(c_size);

            float value = lambda(Vec2{ px, py }, Vec2{ 0.0f, 0.0f });
            value = LinearTosRGB(value);

            if (x == 0 || y == 0 || x == c_size - 1 || y == c_size - 1)
                value = 0;

            *pixel = (unsigned char)Clamp(value * 256.0f, 0.0f, 255.0f);
            pixel++;
        }
    }

    stbi_write_png(outputFileName, c_size, c_size, 1, pixels.data(), 0);
}

template <typename LAMBDA>
void Test(const char* shapeName, float actualValue, const LAMBDA& lambda /*, Vec2 sampleOffset, int indexOffset*/)
{
    printf("===== %s =====\n", shapeName);

    // write the sample counts into the csv
    CSV results;
    SetCSV(results, 0, 0, "Samples");
    for (size_t index = 1; index <= c_numSamples; ++index)
        SetCSV(results, 0, index, "%zu", index);

    // for each type of sampling
    for (const Samples& samples : g_samples)
    {
        // do the tests
        std::vector<float> absError;
        std::vector<float> sqAbsError;
        int lastPercent = -1;
        for (size_t testIndex = 0; testIndex < c_numTests; ++testIndex)
        {
            int percent = int(100.0f * float(testIndex) / float(c_numTests));
            if (percent != lastPercent)
            {
                printf("\r%s: %i%%", samples.name, percent);
                lastPercent = percent;
            }
            DoTest(lambda, actualValue, samples, absError, sqAbsError, testIndex);
        }
        printf("\r%s: 100%%\n", samples.name);

        // if not progressive, only report the final error amount
        if (!samples.progressive)
        {
            float lastAbsError = *absError.rbegin();
            float lastSqAbsError = *sqAbsError.rbegin();
            for (float& f : absError)
                f = lastAbsError;
            for (float& f : sqAbsError)
                f = lastSqAbsError;
        }

        // write to the csv
        size_t col = GetCSVCols(results);
        SetCSV(results, col, 0, "%s", samples.name);
        //SetCSV(results, col + 1, 0, "%s var", samples.name);
        for (size_t index = 0; index < c_numSamples; ++index)
        {
            SetCSV(results, col, index + 1, "%f", absError[index]);

            //float variance = abs(sqAbsError[index] - absError[index] * absError[index]);
            //float stddev = sqrt(variance);
            //SetCSV(results, col, index + 1, "%f", variance);
        }
    }

    // write the output csv and integrand image
    char fileName[1024];
    sprintf(fileName, "out/%s.png", shapeName);
    MakeIntegrandImage(lambda, fileName);
    sprintf(fileName, "out/%s.csv", shapeName);
    WriteCSV(results, fileName);

    // run the python script
    sprintf(fileName, "python MakeGraphs.py %s", shapeName);
    system(fileName);
}

// ====================================================================

int main(int argc, char** argv)
{
    _mkdir("out");

    Test("gaussian", c_GaussianActualValue, Gaussian);
    Test("triangle", c_TriangleActualValue, Triangle);
    Test("step", c_StepActualValue, Step);

    return 0;
}


/*
TODO:
* Test() has the beginnings for passing in spatial and index offsets.
* seed R2? although i guess the others don't use the seed either so...?? what do they do for seeds
* could also offset over index instead of just over distance
* also offset the sequences toroidally.
* report RMSE
* clean up this file, you copy/pasted this from unrelated code
*/