import unittest
import pet_crude_ref_emis
import os
from pet_crude_ref_emis import inventory_workbook_path


class Testget_pet_crude_ref_inv_data(unittest.TestCase):
    def setUp(self):
        self.input_file = inventory_workbook_path
        self.output_file = "pet_crude_ref_emi.csv"

    def tearDown(self) -> None:
        try:
            os.remove(self.output_file)
        except FileNotFoundError:
            pass

    def test_get_pet_crude_ref_inv_data(self):
        pet_crude_ref_emis.get_pet_crude_ref_inv_data(
            self.input_file,
            self.output_file
        )
        self.assertTrue(os.path.exists(self.output_file))


if __name__ == '__main__':
    unittest.main()
