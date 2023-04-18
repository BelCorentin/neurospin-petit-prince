 import getpass
import typing as tp
from . import api
from . import utils


class Pallier2023Recording(api.Recording):

    data_url = ""
    paper_url = ""
    doi = ""
    licence = ''
    modality = "audio"
    language = "fr"
    device = "meg"
    description = "52 subjects listen to a 90min story in French (Le Petit Prince)"

    @classmethod
    def download(cls) -> None:
        paths = utils.StudyPaths(cls.study_name())
        done = paths.download / "done.txt"
        if done.exists():
            print(f"{done} already exists, delete it to redownload")
            return
        paths.download.mkdir(exist_ok=True, parents=True)
        try:
            import webdav3.client as wc  # pylint: disable=import-outside-toplevel
            # see doc https://pypi.org/project/webdavclient3/
        except ImportError as e:
            string = f"Download of {cls.study_name()} requires `pip install webdavclient3`"
            raise RuntimeError(string) from e
        string = """\nYou need an account to bioproj.cea.fr for downloading Pallier2023 study,
if you do not have one, please contact christophe.pallier@cea.fr"""
        print(string)
        usr = input("Username: ")
        pwd = getpass.getpass()
        options = {
            'webdav_hostname': f"https://bioproj.cea.fr/nextcloud/remote.php/dav/files/{usr}/",
            'webdav_login': usr,
            'webdav_password': pwd,
        }
        client = wc.Client(options)
        folder = "LPP_MEG"
        if client.check(folder):
            print("Logged in, and found the dataset. Starting to download")
        else:
            print("Could not find the folder, you may have a login issue")
        client.pull(remote_directory=folder, local_directory=str(paths.download), )
        done.write_text("The presence of this file means the dataset has been "
                        "downloaded, delete it to redownload")

    @classmethod
    def iter(cls) -> tp.Iterator["Pallier2023Recording"]:  # type: ignore
        """Returns a generator of all recordings"""
        # download, extract, organize
        cls.download()  # download if not already downloaded
        # TODO: all the rest
        # for session in ("exp1", "exp2"):
        #     folder = paths.preprocessed / session
        #     for f in folder.iterdir():
        #         if f.name.endswith(".eeg"):
        #             uid = f.name.strip(".eeg")
        #             recording = cls(uid, session=session)
        #             yield recording