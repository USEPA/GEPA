
import unittest
import comb_mobile_emis
import os
from comb_mobile_emis import inventory_workbook_path


class Testget_comb_mobile_inv_data(unittest.TestCase):
    def setUp(self):
        self.input_file = inventory_workbook_path
        self.output_file_comb_mob_ldt_emi = "test_comb_mob_ldt_emi.csv"
        self.output_file_comb_mob_loco_emi = "test_comb_mob_loco_emi.csv"
        self.output_file_comb_mob_mcycle_emi = "test_comb_mob_mcycle_emi.csv"
        self.output_file_comb_mob_other_emi = "test_comb_mob_other_emi.csv"
        self.output_file_comb_mob_pass_emi = "test_comb_mob_pass_emi.csv"

    def tearDown(self) -> None:
        try:
            os.remove(self.output_file_comb_mob_ldt_emi)
            os.remove(self.output_file_comb_mob_loco_emi)
            os.remove(self.output_file_comb_mob_mcycle_emi)
            os.remove(self.output_file_comb_mob_other_emi)
            os.remove(self.output_file_comb_mob_pass_emi)
        except FileNotFoundError:
            pass

    def test_get_comb_mobile_inv_data(self):
        comb_mobile_emis.get_comb_mobile_inv_data(
            self.input_file,
            self.output_file_comb_mob_ldt_emi,
            "comb_mob_ldt_emi"
        )
        self.assertTrue(os.path.exists(self.output_file_comb_mob_ldt_emi))
        comb_mobile_emis.get_comb_mobile_inv_data(
            self.input_file,
            self.output_file_comb_mob_loco_emi,
            "comb_mob_loco_emi"
        )
        self.assertTrue(os.path.exists(self.output_file_comb_mob_loco_emi))
        comb_mobile_emis.get_comb_mobile_inv_data(
            self.input_file,
            self.output_file_comb_mob_mcycle_emi,
            "comb_mob_mcycle_emi"
        )
        self.assertTrue(os.path.exists(self.output_file_comb_mob_mcycle_emi))
        comb_mobile_emis.get_comb_mobile_inv_data(
            self.input_file,
            self.output_file_comb_mob_other_emi,
            "comb_mob_other_emi"
        )
        self.assertTrue(os.path.exists(self.output_file_comb_mob_other_emi))
        comb_mobile_emis.get_comb_mobile_inv_data(
            self.input_file,
            self.output_file_comb_mob_pass_emi,
            "comb_mob_pass_emi"
        )
        self.assertTrue(os.path.exists(self.output_file_comb_mob_pass_emi))


if __name__ == '__main__':
    unittest.main()
