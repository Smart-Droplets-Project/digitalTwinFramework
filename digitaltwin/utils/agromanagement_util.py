import datetime
import yaml


class AgroManagement:
    def __init__(self, agro_management: list):
        self.agro_structure = agro_management
        self.campaign_date: datetime.date = list(agro_management[0].keys())[0]
        self.crop_name: str = agro_management[0][self.campaign_date]['CropCalendar']['crop_name']
        self.crop_variety: str = agro_management[0][self.campaign_date]['CropCalendar']['variety_name']
        self.crop_start_date: datetime.date = agro_management[0][self.campaign_date]['CropCalendar']['crop_start_date']
        self.crop_start_type: str = agro_management[0][self.campaign_date]['CropCalendar']['crop_start_type']
        self.crop_end_date: datetime.date = agro_management[0][self.campaign_date]['CropCalendar']['crop_end_date']
        self.crop_end_type: str = agro_management[0][self.campaign_date]['CropCalendar']['crop_end_type']
        self.max_duration: int = agro_management[0][self.campaign_date]['CropCalendar']['max_duration']

        self.structure = None
        self.build_structure()

    def build_structure(self):
        self.structure = yaml.load(f'''
                    - {self.campaign_date}:
                        CropCalendar:
                            crop_name: {self.crop_name}
                            variety_name: {self.crop_variety}
                            crop_start_date: {self.crop_start_date}
                            crop_start_type: {self.crop_start_type}
                            crop_end_date: {self.crop_end_date}
                            crop_end_type: {self.crop_end_type}
                            max_duration: {self.max_duration}
                        TimedEvents: null
                        StateEvents: null
                ''', Loader=yaml.SafeLoader)

    def set_start_date(self, start_date: datetime.date):
        self.crop_start_date = self.crop_start_date.replace(year=start_date.year,
                                                            month=start_date.month,
                                                            day=start_date.day)

        self.build_structure()

    def set_end_date(self, end_date: datetime.date):
        self.crop_end_date = self.crop_end_date.replace(year=end_date.year,
                                                        month=end_date.month,
                                                        day=end_date.day)

        self.build_structure()

    def set_start_type(self, start):
        assert start == 'sowing' or start == 'emergence'
        self.crop_start_type = start

        self.build_structure()

    def set_end_type(self, end):
        assert end == 'maturity' or end == 'harvest'
        self.crop_end_type = end

        self.build_structure()

    def set_crop_name(self, name='winterwheat'):
        self.crop_name = name

        self.build_structure()

    def set_variety_name(self, name='Arminda'):
        self.crop_variety = name

        self.build_structure()

    def start_sowing(self):
        if self.campaign_date.year == self.crop_end_date.year:
            self.campaign_date = datetime.date(self.crop_end_date.year - 1, 10, 1)
            self.crop_start_date = datetime.date(self.crop_end_date.year - 1, 10, 1)

        self.build_structure()

    def start_emergence(self):
        self.campaign_date = datetime.date(self.crop_end_date.year, 1, 1)
        self.crop_start_date = datetime.date(self.crop_end_date.year, 1, 1)

        self.build_structure()

    def get_start_type(self, start_type):
        self.start_emergence() if start_type == 'emergence' else self.start_sowing()

    def set_max_duration(self, duration):
        self.max_duration = duration

        self.build_structure()

    @property
    def load_agromanagement_file(self):
        return self.structure

    @property
    def get_start_date(self):
        return self.crop_start_date

    @property
    def get_end_date(self):
        return self.crop_end_date