void prepare_dataset_from_root_file(){

	TChain* signal_data_with_magnet = new TChain("calorimeter_data");
	TChain* backgr_data_with_magnet = new TChain("calorimeter_data");
	signal_data_with_magnet->Add("1dot3X0_Charmonium_All_data_root_files/Signal_data_for_electron_positron_energy_momentum_ratio.root");
	backgr_data_with_magnet->Add("1dot3X0_Soft_QCD_data_root_files/Calorimeter_energy_edeption_data_0.root");

	std::vector<std::vector<double>>* calorimeter_cells =
        new std::vector<std::vector<double>>();
	std::vector<std::vector<double>>* particles_in_event =
        new std::vector<std::vector<double>>();

	signal_data_with_magnet->SetBranchAddress("cell_energy", &calorimeter_cells);
    signal_data_with_magnet->SetBranchAddress("event_data", &particles_in_event);

	ofstream myfile;
	myfile.open("../dataset/calorimeter_event_data.txt");

	int c = 0;

	for (int i = 0; i < signal_data_with_magnet->GetEntries(); ++i){
		signal_data_with_magnet->GetEntry(i);
		for (int j = 0; j < particles_in_event->size(); ++j){
			for (auto energy : (*calorimeter_cells)[0]){
				myfile << energy << " ";
			}
			myfile << (*particles_in_event)[j][0] << " ";
	        	if (fabs((*particles_in_event)[j][4]) == 11 or 
					fabs((*particles_in_event)[j][0]) == 22){
				myfile << 0 << "\n";	
			} else {
				myfile << 1 << "\n";
			}
			c++;
		}
		if (c >= 10000){
			break;
		}

	}

	backgr_data_with_magnet->SetBranchAddress("cell_energy", &calorimeter_cells);
    backgr_data_with_magnet->SetBranchAddress("event_data", &particles_in_event);

	c = 0;

	for (int i = 0; i < backgr_data_with_magnet->GetEntries(); ++i){
		backgr_data_with_magnet->GetEntry(i);
		for (int j = 0; j < particles_in_event->size(); ++j){
			for (auto energy : (*calorimeter_cells)[0]){
				myfile << energy << " ";
			}
			myfile << (*particles_in_event)[j][0] << " ";
	        	if (fabs((*particles_in_event)[j][4]) == 11 or 
					fabs((*particles_in_event)[j][0]) == 22){
				myfile << 0 << "\n";	
			} else {
				myfile << 1 << "\n";
			}
			c++;
		}
		if (c >= 10000){
			break;
		}

	}
	myfile.close();

	return;
}
