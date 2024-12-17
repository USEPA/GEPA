import unittest
import ww_emis
import os
from ww_emis import inventory_workbook_path


class Testget_ww_inv_data(unittest.TestCase):
    def setUp(self):
        self.input_file = inventory_workbook_path
        self.output_file_ww_brew = "test_ww_brew_emi.csv"
        self.output_file_ww_ethanol = "test_ww_ethanol_emi.csv"
        self.output_file_ww_petrref = "test_ww_petrref_emi.csv"
        self.output_file_ww_fv = "test_ww_fv_emi.csv"
        self.output_file_ww_mp = "test_ww_mp_emi.csv"
        self.output_file_ww_pp = "test_ww_pp_emi.csv"

    def tearDown(self) -> None:
        try:
            os.remove(self.output_file_ww_brew)
            os.remove(self.output_file_ww_ethanol)
            os.remove(self.output_file_ww_petrref)
            os.remove(self.output_file_ww_fv)
            os.remove(self.output_file_ww_mp)
            os.remove(self.output_file_ww_pp)
        except FileNotFoundError:
            pass

    def test_get_ww_inv_data(self):
        ww_emis.get_ww_inv_data(
            self.input_file,
            self.output_file_ww_brew,
            "ww_brew_emi"
        )
        self.assertTrue(os.path.exists(self.output_file_ww_brew))
        ww_emis.get_ww_inv_data(
            self.input_file,
            self.output_file_ww_ethanol,
            "ww_ethanol_emi"
        )
        self.assertTrue(os.path.exists(self.output_file_ww_ethanol))
        ww_emis.get_ww_inv_data(
            self.input_file,
            self.output_file_ww_fv,
            "ww_fv_emi"
        )
        self.assertTrue(os.path.exists(self.output_file_ww_fv))
        ww_emis.get_ww_inv_data(
            self.input_file,
            self.output_file_ww_mp,
            "ww_mp_emi"
        )
        self.assertTrue(os.path.exists(self.output_file_ww_mp))
        ww_emis.get_ww_inv_data(
            self.input_file,
            self.output_file_ww_petrref,
            "ww_petrref_emi"
        )
        self.assertTrue(os.path.exists(self.output_file_ww_petrref))
        ww_emis.get_ww_inv_data(
            self.input_file,
            self.output_file_ww_pp,
            "ww_pp_emi"
        )
        self.assertTrue(os.path.exists(self.output_file_ww_pp))


if __name__ == '__main__':
    unittest.main()
