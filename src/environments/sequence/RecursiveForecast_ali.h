#ifndef RecursiveForecast_h
#define RecursiveForecast_h

#include <TaskEnv.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <limits>
#include <algorithm>
#include <iostream>

using namespace std;

class RecursiveForecast : public TaskEnv {
  public:
   // Data. One vec per timestep. Supports multivariate
   vector<vector<double>> data;
   // Starting points to slice datasets. One vec each for train, val, test
   vector<vector<int>> t_start;

   int n_prime_;
   // Number of samples (horizon) for training, validation, test
   int n_predict_[3];

   string task_;

   vector<int> uniq_discrete_univars_;

   // Add this declaration
   vector<pair<int, int>> song_indices;

       public:
        CSVReader(const string& f, const string& dlm = " ") : filename(f), delim(dlm) {}
        vector<vector<double>> ReadData() {
            ifstream file(filename);
            vector<vector<double>> data_vec;
            string line;
            while (getline(file, line)) {
                if (line.size() == 0) continue;
                vector<double> row_values;
                stringstream s(line);
                string word;
                while (getline(s, word, ',')) {
                    row_values.push_back(stod(word.c_str()));
                }
                data_vec.push_back(row_values);
            }
            data_vec.push_back(row_values);
         }
         file.close();
         return data_vec;
      }
   };

   RecursiveForecast(string task) {
        eval_type_ = "RecursiveForecast";
        task_ = task;
    }

   ~RecursiveForecast() {}

   int GetNumEval(int phase) { return static_cast<int>(t_start[phase].size()); }

    void PrepareData(mt19937& rng) {
        // import data
        CSVReader* reader;
        if (task_ == "Sunspots")
            reader =
                new CSVReader("./datasets/SN_ms_tot_V2.0_Nov1834-June1926.csv");
        else if (task_ == "Mackey")
            reader = new CSVReader("./datasets/Mackey-1100.csv");
        else if (task_ == "Laser")
            reader = new CSVReader("./datasets/Laser-10000-1000-2100.csv");
        else if (task_ == "Audio")
            reader = new CSVReader("./datasets/2024-05-22-ali-10sec.dat");
        else if (task_ == "Offset")
            reader = new CSVReader("./datasets/ali_offset_diff.csv");
        else if (task_ == "Duration")
            reader = new CSVReader("./datasets/ali_duration.csv");
        else if (task_ == "Pitch")
            reader = new CSVReader("./datasets/ali_pitch.csv");
        else if (task_ == "PitchBach")
            reader = new CSVReader("./datasets/ali_bach_pitch.csv");
        else if (task_ == "Bach")
            reader = new CSVReader(
                "./datasets/input_bach_first_order_offset.csv", ",");
        else {
            cerr << "Unrecognised RecursiveForecast task" << endl;
            exit(1);
        }
        data = reader->ReadData();
        delete reader;

      // Start steps for priming, for each phase (train, validate, test)
      t_start.resize(3);

        if (task_ == "Sunspots" || task_ == "Mackey" || task_ == "Laser") {
            // train (original, 19 start points)
            for (int s = 0; s <= 900; s += 50)
                t_start[0].push_back(s);

            // validation (original, 9 start points)
            for (int s = 50; s <= 850; s += 100)
                t_start[1].push_back(s);

         t_start[_TEST_PHASE].insert(t_start[_TEST_PHASE].begin(), {950});

      } else if (task_ == "Offset" || task_ == "Duration" || task_ == "Pitch" ||
                 task_ == "PitchBach" || task_ == "Bach" ||
                 task_ == "one_song_hot") {
         uniform_int_distribution<int> DisTrain(
             0, data.size() - (n_prime_ + n_predict_[0]));
         uniform_int_distribution<int> DisVal(
             0, data.size() - (n_prime_ + n_predict_[1]));

         // Train
         for (int i = 0; i < n_eval_train_; i++)
            t_start[_TRAIN_PHASE].push_back(DisTrain(rng));

         // Validation
         for (int i = 0; i < n_eval_validation_; i++)
            t_start[_VALIDATION_PHASE].push_back(DisVal(rng));

         // Test
         t_start[_TEST_PHASE] = t_start[_VALIDATION_PHASE];  // test == val

      } else if (task_ == "one_hot_music" || task_ == "uni_variate_cont") {
         vector<int> song_order(task_ == "one_hot_music" ? 30 : 29);
         
         iota(song_order.begin(), song_order.end(), 0);
         // shuffle(song_order.begin(), song_order.end(), rng); //TODO(Ali) - do
         // we want songs in a loop or at random?

         vector<int> train_song_indices(song_order.begin(),
                                        song_order.begin() + 22);
         vector<int> val_song_indices(song_order.begin() + 22,
                                      song_order.begin() + 26);
         vector<int> test_song_indices(song_order.begin() + 26,
                                       song_order.end());

         // Generate random starting points for each phase
         GenerateRandomStarts(song_indices, train_song_indices, _TRAIN_PHASE,
                              n_eval_train_, n_predict_[0], rng);
         cout << "Training Phase Slices: " << endl;
         for (auto i : t_start[_TRAIN_PHASE]) {
            cout << " " << i;
         }

         GenerateRandomStarts(song_indices, val_song_indices, _VALIDATION_PHASE,
                              n_eval_validation_, n_predict_[1], rng);
         cout << "Validation Phase Slices: " << endl;
         for (auto i : t_start[_VALIDATION_PHASE]) {
            cout << " " << i;
         }
         cout << endl;
         GenerateRandomStarts(song_indices, test_song_indices, _TEST_PHASE,
                              n_eval_test_, n_predict_[2], rng);
         cout << "Testing Phase Slices: " << endl;
         for (auto i : t_start[_TEST_PHASE]) {
            cout << " " << i;
         }
         cout << endl;
      } else {
         std::cerr << "Unrecognised task" << std::endl;
         exit(1);
      }

      cout << "time series train slices: " << t_start[_TRAIN_PHASE].size()
           << endl;
      cout << "time series validation slices: "
           << t_start[_VALIDATION_PHASE].size() << endl;
      cout << "time series test slices: " << t_start[_TEST_PHASE].size()
           << endl;
   }

   void GenerateRandomStarts(const vector<pair<int, int>>& song_indices,
                             const vector<int>& song_indices_phase, int phase,
                             int n_eval, int n_predict, mt19937& rng) {
      size_t song_count = song_indices_phase.size();
      t_start[phase].clear();

      int generated_evals = 0;

      while (generated_evals < n_eval) {
         size_t song_idx = song_indices_phase[generated_evals % song_count];

         int start = song_indices[song_idx].first;
         int end = song_indices[song_idx].second;

         int max_start = end - (n_prime_ + n_predict);
         if (max_start <= 0) {
            std::cerr << "Invalid range: " << song_idx << ". Skipping."
                      << std::endl;
            continue;
         }

         std::uniform_int_distribution<int> start_dis(start, max_start - 1);

         std::unordered_set<int> existing_starts(t_start[phase].begin(),
                                                 t_start[phase].end());

         bool unique = false;
         int new_start;
         int attempts = 0;

         while (!unique) {
            new_start = start_dis(rng);
            if (existing_starts.find(new_start) == existing_starts.end()) {
               unique = true;
               t_start[phase].push_back(new_start);
               existing_starts.insert(new_start);
            }
            attempts++;
         }
         if (unique) generated_evals++;
      }
   }

   // Normalize data in each column to the range [0,1]
   void Normalize() {
      size_t n_col = data[0].size();
      for (size_t col = 0; col < n_col; col++) {
         double max_feature = numeric_limits<double>::lowest();
         double min_feature = numeric_limits<double>::max();
         // Find min/max values
         for (size_t row = 0; row < data.size(); row++) {
            max_feature = max(max_feature, data[row][col]);
            min_feature = min(min_feature, data[row][col]);
         }
         // Normalize
         for (size_t row = 0; row < data.size(); row++) {
            data[row][col] =
                (data[row][col] - min_feature) / (max_feature - min_feature);
         }
      }
   }

   void NormalizeSongs() {
      size_t n_col = data[0].size();
      for (size_t col = 0; col < n_col; col++) {
         double max_feature = numeric_limits<double>::lowest();
         double min_feature = numeric_limits<double>::max();

         for (size_t row = 0; row < data.size(); row++) {
            max_feature = max(max_feature, data[row][col]);
            min_feature = min(min_feature, data[row][col]);
         }

         for (size_t row = 0; row < data.size(); row++) {
            data[row][col] =
                (data[row][col] - min_feature) / (max_feature - min_feature);
         }
      }
   }
};

#endif
