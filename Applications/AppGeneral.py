from Generals.ReconGenerals import ReconOption
from Generals.ScannerGenerals import ScannerOption


class AppGeneral:
    def __init__(self, scanner_option: ScannerOption, recon_option: ReconOption):
        self.scanner_option = scanner_option
        self.recon_option = recon_option

    def return_details(self):
        self.scanner_option.return_details()
        self.recon_option.return_details()