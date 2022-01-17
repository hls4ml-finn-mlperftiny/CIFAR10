//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/myproject.h"
#include "firmware/nnet_utils/nnet_helpers.h"

#include <string>

std::vector<std::vector<float>> gen_match_matrix(std::vector<int> preds, std::vector<std::string> ids, int num_tabs=0, bool print=false) {
    std::vector<float> zeros (preds.size(), 0.0);
    std::vector<std::vector<float>> m (preds.size(), zeros);
    
    //S: Define string variables
    std::string tabs = "";
	std::string col_id = "      gt    hp    k     hc  ";
	std::string h_line = "   -------------------------";
	std::string row_x  = "gt |  -  |  -  |  -  |  -  |";
    int col_size = 6;
    int col1_center = 6;

    if (print) {
        //S: Generate tab strings
        for (int i = 0; i < num_tabs; i++) {
            tabs += "\t";
        }

        //S: Print new line/flush output buffer
        std::cout << std::endl;

        //S: Print column id header for matrix
        for (int i = 0; i < ids.size(); i++) {
            col_id.replace(6+(6*i), 2, ids[i]);
        }
        std::cout << std::endl;
        std::cout << tabs << col_id << std::endl;
    }

    //S: Print rows of matrix
    for (int i = 0; i < preds.size(); i++) {
        if (print) {
            std::cout << tabs << h_line << std::endl;
            row_x.replace(0, 2, ids[i]);
        }
        for (int j = 0; j < preds.size(); j++) {
            if (preds[i] == preds[j]) {
                if (print) row_x.replace(col1_center+(col_size*j), 1, "T");
                m[i][j] = 1.0;
            } else {
                if (print) row_x.replace(col1_center+(col_size*j), 1, " ");
            }
		}
        if (print) std::cout << tabs << row_x << std::endl;
	}
    if (print) std::cout << tabs << h_line << std::endl;

    return m;
}

//hls-fpga-machine-learning insert bram

#define CHECKPOINT 5

namespace nnet {
    bool trace_enabled = true;
    std::map<std::string, void *> *trace_outputs = NULL;
    size_t trace_type_size = sizeof(double);
}

int main(int argc, char **argv)
{
  //load input data from text file
  std::ifstream fin("tb_data/tb_input_features.dat");
  //load ground truth class labels from text file
  std::ifstream fpr("tb_data/tb_output_predictions.dat");
  //load hls predictions from text file
  std::ifstream fpr_hls("tb_data/tb_hls_predictions.dat");
  //load keras predictions from text file
  std::ifstream fpr_keras("tb_data/tb_keras_predictions.dat");

#ifdef RTL_SIM
  std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
  std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
  std::ofstream fout(RESULTS_LOG);

  std::string iline;
  std::string pline;
  std::string pline_hls;
  std::string pline_keras;
  int e = 0;

  //Variables to keep track of statistics
  bool class_match = false;
  int total_inputs  = 0;
  int total_matches = 0;
  std::vector<float> zeros (4, 0.0);
  std::vector<std::vector<float>> total_m (4, zeros);
  std::vector<std::vector<float>> curr_m (4, zeros);
  std::vector<std::string> ids = {"gt", "hp", "k ", "hc"};

  if (fin.is_open() && fpr.is_open() && fpr_hls.is_open() && fpr_keras.is_open()) {
    while ( std::getline(fin,iline) && std::getline(fpr,pline) && std::getline(fpr_hls,pline_hls) && std::getline(fpr_keras,pline_keras)) {
      if (e % CHECKPOINT == 0) {
          std::cout << "Processing input " << e << ":"<< std::endl;
      } 
      //Loading input
      char* cstr=const_cast<char*>(iline.c_str());
      char* current;
      std::vector<float> in;
      current=strtok(cstr," ");
      while(current!=NULL) {
        in.push_back(atof(current));
        current=strtok(NULL," ");
      }

      //Load ground truth class label into vector
      cstr=const_cast<char*>(pline.c_str());
      std::vector<float> pr;
      current=strtok(cstr," ");
      while(current!=NULL) {
        pr.push_back(atof(current));
        current=strtok(NULL," ");
      }

      //Load hls prediction into vector
      cstr=const_cast<char*>(pline_hls.c_str());
      std::vector<float> pr_hls;
      current=strtok(cstr," ");
      while(current!=NULL) {
        pr_hls.push_back(atof(current));
        current=strtok(NULL," ");
      }

      //Load keras prediction into vector
      cstr=const_cast<char*>(pline_keras.c_str());
      std::vector<float> pr_keras;
      current=strtok(cstr," ");
      while(current!=NULL) {
        pr_keras.push_back(atof(current));
        current=strtok(NULL," ");
      }

      //hls-fpga-machine-learning insert data
      hls::stream<input_t> input_1("input_1");
      nnet::copy_data<float, input_t, 0, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1>(in, input_1);
      hls::stream<layer18_t> layer18_out("layer18_out");

      //hls-fpga-machine-learning insert top-level-function
      unsigned short size_in1,size_out1;
      myproject(input_1,layer18_out,size_in1,size_out1);

      if (e % CHECKPOINT == 0) {
        std::cout << "\tGround Truth Predictions (gt):\n\t\t" ;//<< std::endl;
        //hls-fpga-machine-learning insert predictions
        for(int i = 0; i < N_LAYER_18; i++) {
          std::cout << pr[i] << " ";
        }

        std::cout << std::endl;
        std::cout << "\thls_py Predictions (hp):\n\t\t" ;//<< std::endl;
        //hls-fpga-machine-learning insert predictions
        for(int i = 0; i < N_LAYER_18; i++) {
          std::cout << pr_hls[i] << " ";
        }

        std::cout << std::endl;
        std::cout << "\tkeras Predictions (k):\n\t\t" ;//<< std::endl;
        //hls-fpga-machine-learning insert predictions
        for(int i = 0; i < N_LAYER_18; i++) {
          std::cout << pr_keras[i] << " ";
        }

        std::cout << std::endl;
        std::cout << "\thls_csim Predictions (hc):\n\t\t" ;//<< std::endl;
        //hls-fpga-machine-learning insert quantized
        nnet::print_result<layer18_t, N_LAYER_18>(layer18_out, std::cout, true);
      }
      e++;


      //hls-fpga-machine-learning insert tb-output
      nnet::print_result<layer18_t, N_LAYER_18>(layer18_out, fout, true);

      //Determine max prediction for both predictions
      layer18_t l18_array = layer18_out.read();
      int pr_max_i  = 0;
      int pr_hls_max_i  = 0;
      int pr_keras_max_i  = 0;
      int qpr_max_i = 0;
      for(int i = 1; i < N_LAYER_18; i++) {

    	  if (pr[i] > pr[pr_max_i]) {
        	pr_max_i = i;
    	  }

    	  if (pr_hls[i] > pr_hls[pr_hls_max_i]) {
        	pr_hls_max_i = i;
    	  }

    	  if (pr_keras[i] > pr_keras[pr_keras_max_i]) {
        	pr_keras_max_i = i;
    	  }


    	  if (l18_array[i] > l18_array[qpr_max_i]) {
    		  qpr_max_i = i;
    	  }
        }

      //Update statistic variables
      class_match = pr_max_i == qpr_max_i;
      total_matches += class_match ? 1 : 0;
      total_inputs++;

        std::vector<int> preds = {pr_max_i, pr_hls_max_i, pr_keras_max_i, qpr_max_i};


      if (e % CHECKPOINT == 0) {
        //Print class match results
        std::cout << "\tPrediction Match Matrix: ";
      }

      curr_m = gen_match_matrix(preds, ids, 2, (e % CHECKPOINT == 0));
      if (e % CHECKPOINT == 0) std::cout << std::endl;

      //S: Add Prediction Match Matrices
      for (int i = 0; i < total_m.size(); i++) {
          for (int j = 0; j < total_m.size(); j++) {
              total_m[i][j] += curr_m[i][j];
          }
      }
    }

    //Print class matching statistics
    std::cout << "Summary of Test:" << std::endl;
    std::cout << "\tTotal # of Test Inputs = " << total_inputs << std::endl;
    std::cout << "\tTotal # of Matching Predictions = " << total_matches;
    std::cout << " (" << (float)total_matches/total_inputs << ")\n"<< std::endl;
    //S: Add Matrices
    for (int i = 0; i < total_m.size(); i++) {
        for (int j = 0; j < total_m.size(); j++) {
            total_m[i][j] /= total_inputs;
            std::cout << "| " << total_m[i][j];
        }
        std::cout << " |" << std::endl;
    }

    fin.close();
    fpr.close();
  } else {
    std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;

    //hls-fpga-machine-learning insert zero
    hls::stream<input_t> input_1("input_1");
    nnet::fill_zero<input_t, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1>(input_1);
    hls::stream<layer18_t> layer18_out("layer18_out");

    //hls-fpga-machine-learning insert top-level-function
    unsigned short size_in1,size_out1;
    myproject(input_1,layer18_out,size_in1,size_out1);

    //hls-fpga-machine-learning insert output
    nnet::print_result<layer18_t, N_LAYER_18>(layer18_out, std::cout, true);

    //hls-fpga-machine-learning insert tb-output
    nnet::print_result<layer18_t, N_LAYER_18>(layer18_out, fout);

  }

  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}