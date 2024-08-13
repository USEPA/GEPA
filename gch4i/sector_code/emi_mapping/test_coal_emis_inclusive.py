
import unittest
import coal_emis_inclusive
import os
from coal_emis_inclusive import inventory_workbook_path


class Testget_coal_inv_data(unittest.TestCase):
    def setUp(self):
        self.input_file = inventory_workbook_path
        self.output_file_coal_post_surf = "test_coal_post_surf_emi.csv"
        self.output_file_coal_post_under = "test_coal_post_under_emi.csv"
        self.output_file_coal_surf = "test_coal_surf_emi.csv"
        self.output_file_coal_under = "test_coal_under_emi.csv"

    def tearDown(self) -> None:
        try:
            os.remove(self.output_file_coal_post_surf)
            os.remove(self.output_file_coal_post_under)
            os.remove(self.output_file_coal_surf)
            os.remove(self.output_file_coal_under)
        except FileNotFoundError:
            pass

    def test_get_coal_inv_data(self):
        coal_emis_inclusive.get_coal_inv_data(
            self.input_file,
            self.output_file_coal_post_surf,
            "coal_post_surf"
        )
        self.assertTrue(os.path.exists(self.output_file_coal_post_surf))
        coal_emis_inclusive.get_coal_inv_data(
            self.input_file,
            self.output_file_coal_post_under,
            "coal_post_under"
        )
        self.assertTrue(os.path.exists(self.output_file_coal_post_under))
        coal_emis_inclusive.get_coal_inv_data(
            self.input_file,
            self.output_file_coal_surf,
            "coal_surf"
        )
        self.assertTrue(os.path.exists(self.output_file_coal_surf))
        coal_emis_inclusive.get_coal_inv_data(
            self.input_file,
            self.output_file_coal_under,
            "coal_under"
        )
        self.assertTrue(os.path.exists(self.output_file_coal_under))


if __name__ == '__main__':
    unittest.main()
