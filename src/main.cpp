#include <iostream>
#include <utility>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "nmmintrin.h"
#include "immintrin.h"
using namespace std;
using namespace std::chrono;
#define GAP -2
#define MISS -1
#define MATCH 1
#define SSE_SIZE 8
#define AVX256_SIZE 16
#define INF 10000

string reverse(string s)
{
    string new_s = "";

    for (int16_t n = s.length() - 1; n >= 0; n--)
    {
        new_s += s[n];
    }
    return new_s;
}
string Make_CIGAR(const string &s)
{
    int16_t i = 1;
    string cigar = "";
    int16_t counter = 1;
    char c = s[0];
    while (i < s.length())
    {

        if (s[i] == s[i - 1])
        {
            counter++;
            if (i == s.length() - 1)
            {
                cigar += to_string(counter);
                cigar += c;
            }
        }
        else
        {
            cigar += to_string(counter);
            cigar += c;
            counter = 1;
            c = s[i];
            if (i == s.length() - 1)
            {
                cigar += to_string(counter);
                cigar += c;
            }
        }
        i++;
    }
    return cigar;
}
int16_t Similarity(char a, char b)
{
    if (a == 'P' && b == 'P')
    {
        return 0;
    }
    else if ((a == 'P' && b != 'P') || (a != 'P' && b == 'P'))
    {
        return GAP;
    }
    else if (a == b)
    {
        return MATCH;
    }
    else
    {
        return MISS;
    }
}
vector<pair<string, string>> getInput(string file_path)
{
    ifstream infile(file_path);
    string line;
    vector<string> sequences;
    vector<pair<string, string>> sequence_pairs;
    while (getline(infile, line))
    {
        if (line[0] == '>')
        {
            continue;
        }
        else
        {
            sequences.push_back(line);
        }
    }
    for (int16_t i = 0; i < sequences.size() - 1; i++)
    {
        pair<string, string> temp;
        temp.first = sequences[i];
        temp.second = sequences[i + 1];
        sequence_pairs.push_back(temp);
    }
    return sequence_pairs;
}
int16_t extractSSE(__m128i reg, int16_t m)
{
    if (m == 0)
        return _mm_extract_epi16(reg, 0);
    else if (m == 1)
        return _mm_extract_epi16(reg, 1);
    else if (m == 2)
        return _mm_extract_epi16(reg, 2);
    else if (m == 3)
        return _mm_extract_epi16(reg, 3);
    else if (m == 4)
        return _mm_extract_epi16(reg, 4);
    else if (m == 5)
        return _mm_extract_epi16(reg, 5);
    else if (m == 6)
        return _mm_extract_epi16(reg, 6);
    else if (m == 7)
        return _mm_extract_epi16(reg, 7);
    else
        return 0;
}
int16_t extractAVX(__m256i reg, int16_t m)
{
    if (m == 0)
        return _mm256_extract_epi16(reg, 0);
    else if (m == 1)
        return _mm256_extract_epi16(reg, 1);
    else if (m == 2)
        return _mm256_extract_epi16(reg, 2);
    else if (m == 3)
        return _mm256_extract_epi16(reg, 3);
    else if (m == 4)
        return _mm256_extract_epi16(reg, 4);
    else if (m == 5)
        return _mm256_extract_epi16(reg, 5);
    else if (m == 6)
        return _mm256_extract_epi16(reg, 6);
    else if (m == 7)
        return _mm256_extract_epi16(reg, 7);
    else if (m == 8)
        return _mm256_extract_epi16(reg, 8);
    else if (m == 9)
        return _mm256_extract_epi16(reg, 9);
    else if (m == 10)
        return _mm256_extract_epi16(reg, 10);
    else if (m == 11)
        return _mm256_extract_epi16(reg, 11);
    else if (m == 12)
        return _mm256_extract_epi16(reg, 12);
    else if (m == 13)
        return _mm256_extract_epi16(reg, 13);
    else if (m == 14)
        return _mm256_extract_epi16(reg, 14);
    else if (m == 15)
        return _mm256_extract_epi16(reg, 15);
    else
        return 0;
}
//--------------------------------global-------------------------------//
pair<vector<int>, vector<string>> globalAlignScalar(const vector<pair<string, string>> &sequences, int sequence_length, bool show_cigar)
{
    vector<int> scores;
    vector<string> cigars;
    int pairsCount = sequences.size();
    for (int pairIndex = 0; pairIndex < pairsCount; pairIndex++)
    {
        int scoresMatrix[sequence_length + 1][sequence_length + 1];
        int similarityMatrix[sequence_length + 1][sequence_length + 1];
        // matrix initiation
        scoresMatrix[0][0] = 0;
        similarityMatrix[0][0] = 0;
        for (int i = 1; i <= sequence_length; i++)
        {
            scoresMatrix[i][0] = GAP * i;
            scoresMatrix[0][i] = GAP * i;

            similarityMatrix[0][i] = 0;
            similarityMatrix[i][0] = 0;
        }
        // filling out the score matrix
        for (int i = 1; i <= sequence_length; i++)
        {
            for (int j = 1; j <= sequence_length; j++)
            {
                int diagonal = scoresMatrix[i - 1][j - 1];
                int up = scoresMatrix[i - 1][j];
                int left = scoresMatrix[i][j - 1];
                similarityMatrix[i][j] = Similarity(sequences[pairIndex].first[i - 1], sequences[pairIndex].second[j - 1]);
                int best = max(diagonal + similarityMatrix[i][j], up + GAP);
                best = max(best, left + GAP);
                scoresMatrix[i][j] = best;
            }
        }
        // making cigar
        if (show_cigar)
        {
            int i = sequence_length;
            int j = sequence_length;
            string align = "";
            while (i > 0 || j > 0)
            {
                if (i > 0 && j > 0 && scoresMatrix[i][j] == scoresMatrix[i - 1][j - 1] + similarityMatrix[i][j])
                {
                    align += 'M';
                    i--;
                    j--;
                }
                else if (j > 0 && scoresMatrix[i][j] == scoresMatrix[i][j - 1] + GAP)
                {
                    align += 'D';
                    j--;
                }
                else if (i > 0 && scoresMatrix[i][j] == scoresMatrix[i - 1][j] + GAP)
                {
                    align += 'I';
                    i--;
                }
                else
                {
                    break;
                }
            }
            string rev = reverse(align);
            string cigar = Make_CIGAR(rev);

            cigars.push_back(cigar);
            scores.push_back(scoresMatrix[sequence_length][sequence_length]);
        }
        else
        {
            cigars.push_back("-");
            scores.push_back(scoresMatrix[sequence_length][sequence_length]);
        }
    }
    pair<vector<int>, vector<string>> b;
    b.first = scores;
    b.second = cigars;
    return b;
}
pair<vector<int16_t>, vector<string>> globalAlignSSE128(const vector<pair<string, string>> &sequences, int16_t sequence_length, bool show_cigar)
{
    __m128i gap = _mm_set1_epi16(GAP);
    __m128i diagonalSSE;
    __m128i upSSE;
    __m128i leftSSE;
    __m128i similaritiesSSE;
    __m128i temp;
    __m128i best;
    vector<int16_t> scores;
    vector<string> cigars;
    int16_t pairsCount = sequences.size();
    for (int16_t pairIndex = 0; pairIndex < pairsCount - SSE_SIZE; pairIndex += SSE_SIZE)
    {
        __m128i scoresMatrix[sequence_length + 1][sequence_length + 1];
        __m128i similarityMatrix[sequence_length + 1][sequence_length + 1];
        // matrix initiation
        scoresMatrix[0][0] = _mm_setzero_si128();
        similarityMatrix[0][0] = _mm_setzero_si128();

        for (int16_t i = 1; i <= sequence_length; i++)
        {
            scoresMatrix[i][0] = _mm_set1_epi16(GAP * i);
            scoresMatrix[0][i] = _mm_set1_epi16(GAP * i);

            similarityMatrix[0][i] = _mm_setzero_si128();
            similarityMatrix[i][0] = _mm_setzero_si128();
        }

        // filling out the score matrix
        for (int16_t i = 1; i <= sequence_length; i++)
        {
            for (int16_t j = 1; j <= sequence_length; j++)
            {
                diagonalSSE = scoresMatrix[i - 1][j - 1];
                upSSE = scoresMatrix[i - 1][j];
                leftSSE = scoresMatrix[i][j - 1];
                int16_t sim[SSE_SIZE];
                for (int16_t m = 0; m < SSE_SIZE; m++)
                {
                    sim[m] = Similarity(sequences[pairIndex + m].first[i - 1], sequences[pairIndex + m].second[j - 1]);
                }
                similaritiesSSE = similarityMatrix[i][j];

                similarityMatrix[i][j] = _mm_loadu_si128((__m128i *)sim);
                temp = _mm_add_epi16(diagonalSSE, similaritiesSSE);
                diagonalSSE = temp;

                temp = _mm_add_epi16(upSSE, gap);
                upSSE = temp;

                temp = _mm_add_epi16(leftSSE, gap);
                leftSSE = temp;

                temp = _mm_max_epi16(diagonalSSE, upSSE);
                best = _mm_max_epi16(temp, leftSSE);
                scoresMatrix[i][j] = best;
            }
        }
        // making cigar
        if (show_cigar)
        {
            for (int16_t m = 0; m < SSE_SIZE; m++)
            {
                int16_t i = sequence_length;
                int16_t j = sequence_length;
                string align = "";
                while (i > 0 || j > 0)
                {
                    int16_t p = extractSSE(scoresMatrix[i][j], m);
                    if (i > 0 && j > 0 && p == extractSSE(scoresMatrix[i - 1][j - 1], m) + extractSSE(similarityMatrix[i][j], m))
                    {
                        align += 'M';
                        i--;
                        j--;
                    }
                    else if (j > 0 && p == extractSSE(scoresMatrix[i][j - 1], m) + GAP)
                    {
                        align += 'D';
                        j--;
                    }
                    else if (i > 0 && p == extractSSE(scoresMatrix[i - 1][j], m) + GAP)
                    {
                        align += 'I';
                        i--;
                    }
                    else
                    {
                        break;
                    }
                }
                string rev = reverse(align);
                string cigar = Make_CIGAR(rev);

                cigars.push_back(cigar);
                scores.push_back(extractSSE(scoresMatrix[sequence_length][sequence_length], m));
            }
        }
        else
        {
            for (int16_t m = 0; m < SSE_SIZE; m++)
            {
                cigars.push_back("-");
                scores.push_back(extractSSE(scoresMatrix[sequence_length][sequence_length], m));
            }
        }
    }
    pair<vector<int16_t>, vector<string>> b;
    b.first = scores;
    b.second = cigars;
    return b;
}
pair<vector<int16_t>, vector<string>> globalAlignAVX256(const vector<pair<string, string>> &sequences, int16_t sequence_length, bool show_cigar)
{
    __m256i gap = _mm256_set1_epi16(GAP);
    __m256i diagonalSSE;
    __m256i upSSE;
    __m256i leftSSE;
    __m256i similaritiesSSE;
    __m256i temp;
    __m256i best;

    vector<int16_t> scores;
    vector<string> cigars;
    int16_t pairsCount = sequences.size();
    for (int16_t pairIndex = 0; pairIndex < pairsCount - AVX256_SIZE; pairIndex += AVX256_SIZE)
    {
        __m256i scoresMatrix[sequence_length + 1][sequence_length + 1];
        __m256i similarityMatrix[sequence_length + 1][sequence_length + 1];
        // matrix initiation

        scoresMatrix[0][0] = _mm256_setzero_si256();
        similarityMatrix[0][0] = _mm256_setzero_si256();
        for (int16_t i = 1; i <= sequence_length; i++)
        {
            scoresMatrix[i][0] = _mm256_set1_epi16(GAP * i);
            scoresMatrix[0][i] = _mm256_set1_epi16(GAP * i);

            similarityMatrix[0][i] = _mm256_setzero_si256();
            similarityMatrix[i][0] = _mm256_setzero_si256();
        }

        // filling out the score matrix
        for (int16_t i = 1; i <= sequence_length; i++)
        {
            for (int16_t j = 1; j <= sequence_length; j++)
            {
                diagonalSSE = scoresMatrix[i - 1][j - 1];
                upSSE = scoresMatrix[i - 1][j];
                leftSSE = scoresMatrix[i][j - 1];
                int16_t sim[AVX256_SIZE];
                for (int16_t m = 0; m < AVX256_SIZE; m++)
                {
                    sim[m] = Similarity(sequences[pairIndex + m].first[i - 1], sequences[pairIndex + m].second[j - 1]);
                }
                similarityMatrix[i][j] = _mm256_loadu_si256((__m256i *)sim);
                similaritiesSSE = similarityMatrix[i][j];

                temp = _mm256_add_epi16(diagonalSSE, similaritiesSSE);
                diagonalSSE = temp;

                temp = _mm256_add_epi16(upSSE, gap);
                upSSE = temp;

                temp = _mm256_add_epi16(leftSSE, gap);
                leftSSE = temp;

                temp = _mm256_max_epi16(diagonalSSE, upSSE);
                best = _mm256_max_epi16(temp, leftSSE);
                scoresMatrix[i][j] = best;
            }
        }
        // making cigar
        if (show_cigar)
        {
            for (int16_t m = 0; m < AVX256_SIZE; m++)
            {
                int16_t i = sequence_length;
                int16_t j = sequence_length;
                string align = "";
                while (i > 0 || j > 0)
                {
                    int16_t p = extractAVX(scoresMatrix[i][j], m);
                    if (i > 0 && j > 0 && p == extractAVX(scoresMatrix[i - 1][j - 1], m) + extractAVX(similarityMatrix[i][j], m))
                    {
                        align += 'M';
                        i--;
                        j--;
                    }
                    else if (j > 0 && p == extractAVX(scoresMatrix[i][j - 1], m) + GAP)
                    {
                        align += 'D';
                        j--;
                    }
                    else if (i > 0 && p == extractAVX(scoresMatrix[i - 1][j], m) + GAP)
                    {
                        align += 'I';
                        i--;
                    }
                    else
                    {
                        break;
                    }
                }
                string rev = reverse(align);
                string cigar = Make_CIGAR(rev);

                cigars.push_back(cigar);
                scores.push_back(extractAVX(scoresMatrix[sequence_length][sequence_length], m));
            }
        }
        else
        {
            for (int16_t m = 0; m < AVX256_SIZE; m++)
            {
                cigars.push_back("-");
                scores.push_back(extractAVX(scoresMatrix[sequence_length][sequence_length], m));
            }
        }
    }
    pair<vector<int16_t>, vector<string>> b;
    b.first = scores;
    b.second = cigars;
    return b;
}
//--------------------------------local-------------------------------//
pair<vector<int>, vector<string>> localAlignScalar(const vector<pair<string, string>> &sequences, int sequence_length, bool show_cigar)
{
    vector<int> scores;
    vector<string> cigars;
    int pairsCount = sequences.size();
    for (int pairIndex = 0; pairIndex < pairsCount; pairIndex++)
    {
        int scoresMatrix[sequence_length + 1][sequence_length + 1];
        int similarityMatrix[sequence_length + 1][sequence_length + 1];
        // matrix initiation
        scoresMatrix[0][0] = 0;
        similarityMatrix[0][0] = 0;
        for (int i = 1; i <= sequence_length; i++)
        {
            scoresMatrix[i][0] = 0;
            scoresMatrix[0][i] = 0;

            similarityMatrix[0][i] = 0;
            similarityMatrix[i][0] = 0;
        }
        // filling out the score matrix
        for (int i = 1; i <= sequence_length; i++)
        {
            for (int j = 1; j <= sequence_length; j++)
            {
                int diagonal = scoresMatrix[i - 1][j - 1];
                int up = scoresMatrix[i - 1][j];
                int left = scoresMatrix[i][j - 1];
                similarityMatrix[i][j] = Similarity(sequences[pairIndex].first[i - 1], sequences[pairIndex].second[j - 1]);

                int best = max(diagonal + similarityMatrix[i][j], up + GAP);
                best = max(best, left + GAP);
                best = max(best, 0);
                scoresMatrix[i][j] = best;
            }
        }
        // making cigar
        // finding maximum score
        int i = 0;
        int j = 0;
        int maximumScore = 0;
        for (int ii = 0; ii <= sequence_length; ii++)
        {
            for (int jj = 0; jj <= sequence_length; jj++)
            {
                if (scoresMatrix[ii][jj] > maximumScore)
                {
                    maximumScore = scoresMatrix[ii][jj];
                    i = ii;
                    j = jj;
                }
            }
        }
        if (show_cigar)
        {
            string align = "";
            while ((i > 0 || j > 0) && scoresMatrix[i][j] != 0)
            {
                if (i > 0 && j > 0 && scoresMatrix[i][j] == scoresMatrix[i - 1][j - 1] + similarityMatrix[i][j])
                {
                    align += 'M';
                    i--;
                    j--;
                }
                else if (j > 0 && scoresMatrix[i][j] == scoresMatrix[i][j - 1] + GAP)
                {
                    align += 'D';
                    j--;
                }
                else if (i > 0 && scoresMatrix[i][j] == scoresMatrix[i - 1][j] + GAP)
                {
                    align += 'I';
                    i--;
                }
                else
                {
                    break;
                }
            }
            string rev = reverse(align);
            string cigar = Make_CIGAR(rev);

            cigars.push_back(cigar);
            scores.push_back(maximumScore);
        }
        else
        {
            cigars.push_back("-");
            scores.push_back(maximumScore);
        }
    }
    pair<vector<int>, vector<string>> b;
    b.first = scores;
    b.second = cigars;
    return b;
}
pair<vector<int16_t>, vector<string>> localAlignSSE128(const vector<pair<string, string>> &sequences, int16_t sequence_length, bool show_cigar)
{
    __m128i gap = _mm_set1_epi16(GAP);
    __m128i zero = _mm_setzero_si128();
    __m128i diagonalSSE;
    __m128i upSSE;
    __m128i leftSSE;
    __m128i similaritiesSSE;
    __m128i temp;
    __m128i best;
    vector<int16_t> scores;
    vector<string> cigars;
    int16_t pairsCount = sequences.size();
    for (int16_t pairIndex = 0; pairIndex < pairsCount - SSE_SIZE; pairIndex += SSE_SIZE)
    {
        __m128i scoresMatrix[sequence_length + 1][sequence_length + 1];
        __m128i similarityMatrix[sequence_length + 1][sequence_length + 1];
        // matrix initiation
        scoresMatrix[0][0] = _mm_setzero_si128();
        similarityMatrix[0][0] = _mm_setzero_si128();
        for (int16_t i = 1; i <= sequence_length; i++)
        {
            scoresMatrix[i][0] = _mm_setzero_si128();
            scoresMatrix[0][i] = _mm_setzero_si128();

            similarityMatrix[0][i] = _mm_setzero_si128();
            similarityMatrix[i][0] = _mm_setzero_si128();
        }

        // filling out the score matrix
        for (int16_t i = 1; i <= sequence_length; i++)
        {
            for (int16_t j = 1; j <= sequence_length; j++)
            {
                diagonalSSE = scoresMatrix[i - 1][j - 1];
                upSSE = scoresMatrix[i - 1][j];
                leftSSE = scoresMatrix[i][j - 1];
                int16_t sim[SSE_SIZE];
                for (int16_t m = 0; m < SSE_SIZE; m++)
                {
                    sim[m] = Similarity(sequences[pairIndex + m].first[i - 1], sequences[pairIndex + m].second[j - 1]);
                }
                similarityMatrix[i][j] = _mm_loadu_si128((__m128i *)sim);
                similaritiesSSE = similarityMatrix[i][j];

                temp = _mm_add_epi16(diagonalSSE, similaritiesSSE);
                diagonalSSE = temp;

                temp = _mm_add_epi16(upSSE, gap);
                upSSE = temp;

                temp = _mm_add_epi16(leftSSE, gap);
                leftSSE = temp;

                best = _mm_max_epi16(diagonalSSE, upSSE);
                temp = _mm_max_epi16(best, leftSSE);
                best = _mm_max_epi16(temp, zero);
                scoresMatrix[i][j] = best;
            }
        }
        // making cigar

        for (int16_t m = 0; m < SSE_SIZE; m++)
        {
            // finding maximum score
            int16_t i = 0;
            int16_t j = 0;
            int16_t maximumScore = 0;
            for (int16_t ii = 0; ii <= sequence_length; ii++)
            {
                for (int16_t jj = 0; jj <= sequence_length; jj++)
                {
                    int16_t p = extractSSE(scoresMatrix[ii][jj], m);
                    if (p > maximumScore)
                    {
                        maximumScore = p;
                        i = ii;
                        j = jj;
                    }
                }
            }
            if (show_cigar)
            {

                string align = "";
                while ((i > 0 || j > 0) && extractSSE(scoresMatrix[i][j], m) != 0)
                {
                    int16_t p = extractSSE(scoresMatrix[i][j], m);
                    if (i > 0 && j > 0 && p == extractSSE(scoresMatrix[i - 1][j - 1], m) + extractSSE(similarityMatrix[i][j], m))
                    {
                        align += 'M';
                        i--;
                        j--;
                    }
                    else if (j > 0 && p == extractSSE(scoresMatrix[i][j - 1], m) + GAP)
                    {
                        align += 'D';
                        j--;
                    }
                    else if (i > 0 && p == extractSSE(scoresMatrix[i - 1][j], m) + GAP)
                    {
                        align += 'I';
                        i--;
                    }
                    else
                    {
                        break;
                    }
                }
                string rev = reverse(align);
                string cigar = Make_CIGAR(rev);

                cigars.push_back(cigar);
                scores.push_back(maximumScore);
            }
            else
            {

                cigars.push_back("-");
                scores.push_back(maximumScore);
            }
        }
    }
    pair<vector<int16_t>, vector<string>> b;
    b.first = scores;
    b.second = cigars;
    return b;
}
pair<vector<int16_t>, vector<string>> localAlignAVX256(const vector<pair<string, string>> &sequences, int16_t sequence_length, bool show_cigar)
{
    __m256i gap = _mm256_set1_epi16(GAP);
    __m256i zero = _mm256_setzero_si256();
    __m256i diagonalSSE;
    __m256i upSSE;
    __m256i leftSSE;
    __m256i similaritiesSSE;
    __m256i temp;
    __m256i best;
    vector<int16_t> scores;
    vector<string> cigars;
    int16_t pairsCount = sequences.size();
    for (int16_t pairIndex = 0; pairIndex < pairsCount; pairIndex += AVX256_SIZE)
    {
        __m256i scoresMatrix[sequence_length + 1][sequence_length + 1];
        __m256i similarityMatrix[sequence_length + 1][sequence_length + 1];
        // matrix initiation
        scoresMatrix[0][0] = _mm256_setzero_si256();
        similarityMatrix[0][0] = _mm256_setzero_si256();
        for (int16_t i = 1; i <= sequence_length; i++)
        {
            scoresMatrix[i][0] = _mm256_setzero_si256();
            scoresMatrix[0][i] = _mm256_setzero_si256();

            similarityMatrix[0][i] = _mm256_setzero_si256();
            similarityMatrix[i][0] = _mm256_setzero_si256();
        }
        // filling out the score matrix
        for (int16_t i = 1; i <= sequence_length; i++)
        {
            for (int16_t j = 1; j <= sequence_length; j++)
            {
                diagonalSSE = scoresMatrix[i - 1][j - 1];
                upSSE = scoresMatrix[i - 1][j];
                leftSSE = scoresMatrix[i][j - 1];
                int16_t sim[AVX256_SIZE];
                for (int16_t m = 0; m < AVX256_SIZE; m++)
                {
                    sim[m] = Similarity(sequences[pairIndex + m].first[i - 1], sequences[pairIndex + m].second[j - 1]);
                }
                similarityMatrix[i][j] = _mm256_loadu_si256((__m256i *)sim);
                similaritiesSSE = similarityMatrix[i][j];

                temp = _mm256_add_epi16(diagonalSSE, similaritiesSSE);
                diagonalSSE = temp;

                temp = _mm256_add_epi16(upSSE, gap);
                upSSE = temp;

                temp = _mm256_add_epi16(leftSSE, gap);
                leftSSE = temp;

                best = _mm256_max_epi16(diagonalSSE, upSSE);
                temp = _mm256_max_epi16(best, leftSSE);
                best = _mm256_max_epi16(temp, zero);
                scoresMatrix[i][j] = best;
            }
        }
        // making cigar
        for (int16_t m = 0; m < AVX256_SIZE; m++)
        {
            // finding maximum score
            int16_t i = 0;
            int16_t j = 0;
            int16_t maximumScore = 0;
            for (int16_t ii = 0; ii <= sequence_length; ii++)
            {
                for (int16_t jj = 0; jj <= sequence_length; jj++)
                {
                    int16_t p = extractAVX(scoresMatrix[ii][jj], m);
                    if (p > maximumScore)
                    {
                        maximumScore = p;
                        i = ii;
                        j = jj;
                    }
                }
            }
            if (show_cigar)
            {
                string align = "";
                while ((i > 0 || j > 0) && extractAVX(scoresMatrix[i][j], m) != 0)
                {
                    int16_t p = extractAVX(scoresMatrix[i][j], m);
                    if (i > 0 && j > 0 && p == extractAVX(scoresMatrix[i - 1][j - 1], m) + extractAVX(similarityMatrix[i][j], m))
                    {
                        align += 'M';
                        i--;
                        j--;
                    }
                    else if (j > 0 && p == extractAVX(scoresMatrix[i][j - 1], m) + GAP)
                    {
                        align += 'D';
                        j--;
                    }
                    else if (i > 0 && p == extractAVX(scoresMatrix[i - 1][j], m) + GAP)
                    {
                        align += 'I';
                        i--;
                    }
                    else
                    {
                        break;
                    }
                }
                string rev = reverse(align);
                string cigar = Make_CIGAR(rev);

                cigars.push_back(cigar);
                scores.push_back(maximumScore);
            }
            else
            {
                cigars.push_back("-");
                scores.push_back(maximumScore);
            }
        }
    }
    pair<vector<int16_t>, vector<string>> b;
    b.first = scores;
    b.second = cigars;
    return b;
}
//--------------------------------glocal-------------------------------//
pair<vector<int>, vector<string>> glocalAlignScalar(const vector<pair<string, string>> &sequences, int sequence_length, bool show_cigar)
{
    vector<int> scores;
    vector<string> cigars;
    int pairsCount = sequences.size();
    for (int pairIndex = 0; pairIndex < pairsCount; pairIndex++)
    {
        int scoresMatrix[sequence_length + 1][sequence_length + 1];
        int similarityMatrix[sequence_length + 1][sequence_length + 1];
        // matrix initiation
        scoresMatrix[0][0] = 0;
        similarityMatrix[0][0] = 0;
        for (int i = 1; i <= sequence_length; i++)
        {
            scoresMatrix[i][0] = 0;
            scoresMatrix[0][i] = 0;

            similarityMatrix[0][i] = 0;
            similarityMatrix[i][0] = 0;
        }

        // filling out the score matrix
        for (int i = 1; i <= sequence_length; i++)
        {
            for (int j = 1; j <= sequence_length; j++)
            {
                int diagonal = scoresMatrix[i - 1][j - 1];
                int up = scoresMatrix[i - 1][j];
                int left = scoresMatrix[i][j - 1];
                similarityMatrix[i][j] = Similarity(sequences[pairIndex].first[i - 1], sequences[pairIndex].second[j - 1]);
                int best = max(diagonal + similarityMatrix[i][j], up + GAP);
                best = max(best, left + GAP);
                scoresMatrix[i][j] = best;
            }
        }
        // making cigar
        // finding maximum score
        int i = 0;
        int j = 0;
        int maximumScore = -INF;
        for (int ii = 0; ii <= sequence_length; ii++)
        {
            if (scoresMatrix[ii][sequence_length] > maximumScore)
            {
                maximumScore = scoresMatrix[ii][sequence_length];
                i = ii;
                j = sequence_length;
            }
            if (scoresMatrix[sequence_length][ii] > maximumScore)
            {
                maximumScore = scoresMatrix[sequence_length][ii];
                i = sequence_length;
                j = ii;
            }
        }
        if (show_cigar)
        {
            string align = "";
            while (i > 0 && j > 0)
            {
                if (i > 0 && j > 0 && scoresMatrix[i][j] == scoresMatrix[i - 1][j - 1] + similarityMatrix[i][j])
                {
                    align += 'M';
                    i--;
                    j--;
                }
                else if (j > 0 && scoresMatrix[i][j] == scoresMatrix[i][j - 1] + GAP)
                {
                    align += 'D';
                    j--;
                }
                else if (i > 0 && scoresMatrix[i][j] == scoresMatrix[i - 1][j] + GAP)
                {
                    align += 'I';
                    i--;
                }
                else
                {
                    break;
                }
            }
            string rev = reverse(align);
            string cigar = Make_CIGAR(rev);

            cigars.push_back(cigar);
            scores.push_back(maximumScore);
        }
        else
        {
            cigars.push_back("-");
            scores.push_back(maximumScore);
        }
    }
    pair<vector<int>, vector<string>> b;
    b.first = scores;
    b.second = cigars;
    return b;
}
pair<vector<int16_t>, vector<string>> glocalAlignSSE128(const vector<pair<string, string>> &sequences, int16_t sequence_length, bool show_cigar)
{
    __m128i gap = _mm_set1_epi16(GAP);
    __m128i diagonalSSE;
    __m128i upSSE;
    __m128i leftSSE;
    __m128i similaritiesSSE;
    __m128i temp;
    __m128i best;
    vector<int16_t> scores;
    vector<string> cigars;
    int16_t pairsCount = sequences.size();
    for (int16_t pairIndex = 0; pairIndex < pairsCount - SSE_SIZE; pairIndex += SSE_SIZE)
    {
        __m128i scoresMatrix[sequence_length + 1][sequence_length + 1];
        __m128i similarityMatrix[sequence_length + 1][sequence_length + 1];
        // matrix initiation
        scoresMatrix[0][0] = _mm_setzero_si128();
        similarityMatrix[0][0] = _mm_setzero_si128();
        for (int16_t i = 1; i <= sequence_length; i++)
        {
            scoresMatrix[i][0] = _mm_setzero_si128();
            scoresMatrix[0][i] = _mm_setzero_si128();

            similarityMatrix[0][i] = _mm_setzero_si128();
            similarityMatrix[i][0] = _mm_setzero_si128();
        }

        // filling out the score matrix
        for (int16_t i = 1; i <= sequence_length; i++)
        {
            for (int16_t j = 1; j <= sequence_length; j++)
            {
                diagonalSSE = scoresMatrix[i - 1][j - 1];
                upSSE = scoresMatrix[i - 1][j];
                leftSSE = scoresMatrix[i][j - 1];
                int16_t sim[SSE_SIZE];
                for (int16_t m = 0; m < SSE_SIZE; m++)
                {
                    sim[m] = Similarity(sequences[pairIndex + m].first[i - 1], sequences[pairIndex + m].second[j - 1]);
                }
                similarityMatrix[i][j] = _mm_loadu_si128((__m128i *)sim);
                similaritiesSSE = similarityMatrix[i][j];

                temp = _mm_add_epi16(diagonalSSE, similaritiesSSE);
                diagonalSSE = temp;

                temp = _mm_add_epi16(upSSE, gap);
                upSSE = temp;

                temp = _mm_add_epi16(leftSSE, gap);
                leftSSE = temp;

                temp = _mm_max_epi16(diagonalSSE, upSSE);
                best = _mm_max_epi16(temp, leftSSE);
                scoresMatrix[i][j] = best;
            }
        }
        // making cigar

        for (int16_t m = 0; m < SSE_SIZE; m++)
        {
            // finding maximum score
            int16_t i = 0;
            int16_t j = 0;
            int16_t maximumScore = -INF;
            for (int16_t ii = 0; ii <= sequence_length; ii++)
            {
                int16_t p = extractSSE(scoresMatrix[ii][sequence_length], m);
                int16_t q = extractSSE(scoresMatrix[sequence_length][ii], m);

                if (p > maximumScore)
                {
                    maximumScore = p;
                    i = ii;
                    j = sequence_length;
                }
                if (q > maximumScore)
                {
                    maximumScore = q;
                    i = sequence_length;
                    j = ii;
                }
            }
            if (show_cigar)
            {

                string align = "";
                while (i > 0 && j > 0)
                {
                    int16_t p = extractSSE(scoresMatrix[i][j], m);

                    if (i > 0 && j > 0 && p == extractSSE(scoresMatrix[i - 1][j - 1], m) + extractSSE(similarityMatrix[i][j], m))
                    {
                        align += 'M';
                        i--;
                        j--;
                    }
                    else if (j > 0 && p == extractSSE(scoresMatrix[i][j - 1], m) + GAP)
                    {
                        align += 'D';
                        j--;
                    }
                    else if (i > 0 && p == extractSSE(scoresMatrix[i - 1][j], m) + GAP)
                    {
                        align += 'I';
                        i--;
                    }
                    else
                    {
                        break;
                    }
                }
                string rev = reverse(align);
                string cigar = Make_CIGAR(rev);

                cigars.push_back(cigar);
                scores.push_back(maximumScore);
            }
            else
            {

                cigars.push_back("-");
                scores.push_back(maximumScore);
            }
        }
    }
    pair<vector<int16_t>, vector<string>> b;
    b.first = scores;
    b.second = cigars;
    return b;
}
pair<vector<int16_t>, vector<string>> glocalAlignAVX256(const vector<pair<string, string>> &sequences, int16_t sequence_length, bool show_cigar)
{
    __m256i gap = _mm256_set1_epi16(GAP);
    __m256i diagonalSSE;
    __m256i upSSE;
    __m256i leftSSE;
    __m256i similaritiesSSE;
    __m256i temp;
    union
    {
        __m256i best;
        int16_t bests[AVX256_SIZE];
    };

    vector<int16_t> scores;
    vector<string> cigars;
    int16_t pairsCount = sequences.size();
    for (int16_t pairIndex = 0; pairIndex < pairsCount - AVX256_SIZE; pairIndex += AVX256_SIZE)
    {
        __m256i scoresMatrix[sequence_length + 1][sequence_length + 1];
        __m256i similarityMatrix[sequence_length + 1][sequence_length + 1];
        // matrix initiation
        scoresMatrix[0][0] = _mm256_setzero_si256();
        similarityMatrix[0][0] = _mm256_setzero_si256();
        for (int16_t i = 1; i <= sequence_length; i++)
        {
            scoresMatrix[i][0] = _mm256_setzero_si256();
            scoresMatrix[0][i] = _mm256_setzero_si256();

            similarityMatrix[0][i] = _mm256_setzero_si256();
            similarityMatrix[i][0] = _mm256_setzero_si256();
        }

        // filling out the score matrix
        for (int16_t i = 1; i <= sequence_length; i++)
        {
            for (int16_t j = 1; j <= sequence_length; j++)
            {
                diagonalSSE = scoresMatrix[i - 1][j - 1];
                upSSE = scoresMatrix[i - 1][j];
                leftSSE = scoresMatrix[i][j - 1];
                int16_t sim[AVX256_SIZE];
                for (int16_t m = 0; m < AVX256_SIZE; m++)
                {
                    sim[m] = Similarity(sequences[pairIndex + m].first[i - 1], sequences[pairIndex + m].second[j - 1]);
                }
                similarityMatrix[i][j] = _mm256_loadu_si256((__m256i *)sim);
                similaritiesSSE = similarityMatrix[i][j];

                temp = _mm256_add_epi16(diagonalSSE, similaritiesSSE);
                diagonalSSE = temp;

                temp = _mm256_add_epi16(upSSE, gap);
                upSSE = temp;

                temp = _mm256_add_epi16(leftSSE, gap);
                leftSSE = temp;

                temp = _mm256_max_epi16(diagonalSSE, upSSE);
                best = _mm256_max_epi16(temp, leftSSE);
                scoresMatrix[i][j] = best;
            }
        }
        // making cigar
        for (int16_t m = 0; m < AVX256_SIZE; m++)
        {
            // finding maximum score
            int16_t i = 0;
            int16_t j = 0;
            int16_t maximumScore = -INF;
            for (int16_t ii = 0; ii <= sequence_length; ii++)
            {
                int16_t p = extractAVX(scoresMatrix[ii][sequence_length], m);
                int16_t q = extractAVX(scoresMatrix[sequence_length][ii], m);
                if (p > maximumScore)
                {
                    maximumScore = p;
                    i = ii;
                    j = sequence_length;
                }
                if (q > maximumScore)
                {
                    maximumScore = q;
                    i = sequence_length;
                    j = ii;
                }
            }
            if (show_cigar)
            {
                string align = "";
                while (i > 0 && j > 0)
                {
                    int16_t p = extractAVX(scoresMatrix[i][j], m);
                    if (i > 0 && j > 0 && p == extractAVX(scoresMatrix[i - 1][j - 1], m) + extractAVX(similarityMatrix[i][j], m))
                    {
                        align += 'M';
                        i--;
                        j--;
                    }
                    else if (j > 0 && p == extractAVX(scoresMatrix[i][j - 1], m) + GAP)
                    {
                        align += 'D';
                        j--;
                    }
                    else if (i > 0 && p == extractAVX(scoresMatrix[i - 1][j], m) + GAP)
                    {
                        align += 'I';
                        i--;
                    }
                    else
                    {
                        break;
                    }
                }
                string rev = reverse(align);
                string cigar = Make_CIGAR(rev);

                cigars.push_back(cigar);
                scores.push_back(maximumScore);
            }
            else
            {
                cigars.push_back("-");
                scores.push_back(maximumScore);
            }
        }
    }
    pair<vector<int16_t>, vector<string>> b;
    b.first = scores;
    b.second = cigars;
    return b;
}

int main()
{
    vector<pair<string, string>> temp = getInput("Data/sequences.txt");
    ofstream output("docs/c++_results.txt");
    output << "no cigar" << endl;
    int16_t size = 200;

    int result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto a = globalAlignScalar(temp, size, false);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " global scalar" << endl;

    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto tt = globalAlignSSE128(temp, size, false);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " global sse" << endl;

    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto tt = globalAlignAVX256(temp, size, false);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " global avx" << endl;
    // local

    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto a = localAlignScalar(temp, size, false);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " local scalar" << endl;

    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto tt = localAlignSSE128(temp, size, false);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " local sse" << endl;

    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto tt = localAlignAVX256(temp, size, false);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " local avx" << endl;

    // glocal
    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto a = glocalAlignScalar(temp, size, false);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " glocal scalar" << endl;

    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto tt = glocalAlignSSE128(temp, size, false);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " glocal sse" << endl;

    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto tt = glocalAlignAVX256(temp, size, false);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " glocal avx" << endl;

    output << "CIGAR" << endl;

    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto a = globalAlignScalar(temp, size, true);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " global scalar" << endl;

    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto tt = globalAlignSSE128(temp, size, true);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " global sse" << endl;

    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto tt = globalAlignAVX256(temp, size, true);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " global avx" << endl;
    // local
    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto a = localAlignScalar(temp, size, true);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " local scalar" << endl;

    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto tt = localAlignSSE128(temp, size, true);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " local sse " << endl;

    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto tt = localAlignAVX256(temp, size, true);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " local avx " << endl;
    // glocal
    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto a = glocalAlignScalar(temp, size, true);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " glocal scalar" << endl;

    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto tt = glocalAlignSSE128(temp, size, true);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " glocal sse" << endl;

    result = 0;
    for (int counter = 0; counter < 10; counter++)
    {
        auto start = high_resolution_clock::now();
        auto tt = glocalAlignAVX256(temp, size, true);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        result += duration.count();
    }
    output << result / 10 << " glocal avx" << endl;
}