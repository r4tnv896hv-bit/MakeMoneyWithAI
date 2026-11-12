"""Offline tests: no network or API keys needed. Run with `python -m unittest`."""
import csv
import os
import tempfile
import unittest
from unittest import mock

import fetch_projects as fp

ROOT = os.path.dirname(os.path.abspath(__file__))


def make_repo(repo_id, name, description, stars=20000, owner="acme"):
    return {
        "id": repo_id,
        "name": name,
        "full_name": f"{owner}/{name}",
        "owner": {"login": owner},
        "description": description,
        "stargazers_count": stars,
        "html_url": f"https://github.com/{owner}/{name}",
        "archived": False,
    }


class FormatStarsTest(unittest.TestCase):
    def test_units(self):
        self.assertEqual(fp.format_stars(999), "999")
        self.assertEqual(fp.format_stars(1000), "1.0k")
        self.assertEqual(fp.format_stars(453287), "453.3k")
        self.assertEqual(fp.format_stars(1_000_000), "1.0m")


class NeedsBusinessModelTest(unittest.TestCase):
    def test_missing_or_fallback_is_retried(self):
        self.assertTrue(fp.needs_business_model("", "desc"))
        self.assertTrue(fp.needs_business_model("A tool.", "A tool."))
        self.assertTrue(fp.needs_business_model("Old description, no bold", "New description"))

    def test_generated_text_is_kept(self):
        self.assertFalse(fp.needs_business_model("Sell **SaaS** agents.", "A tool."))


class SelectReposTest(unittest.TestCase):
    def test_extras_survive_the_star_cutoff(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = os.getcwd()
        os.chdir(tmp.name)
        self.addCleanup(os.chdir, cwd)
        with open(fp.EXTRA_FILE, "w") as f:
            f.write("Acme/Small\n")
        with open(fp.EXCLUDE_FILE, "w") as f:
            f.write("blocked\n")

        def row(repo_id, name, stars):
            return {"id": repo_id, "owner": "acme", "name": name, "stars": stars,
                    "url": "", "business_model": ""}

        repos = {1: row(1, "big", 300), 2: row(2, "mid", 200), 3: row(3, "low", 100),
                 4: row(4, "small", 5), 5: row(5, "blocked", 999)}
        with mock.patch.object(fp, "MAX_REPOS", 3):
            names = [r["name"] for r in fp.select_repos(repos)]
        self.assertEqual(names, ["big", "mid", "small"])


class RepoDataTest(unittest.TestCase):
    """Consistency checks on the committed data files."""

    def setUp(self):
        self._cwd = os.getcwd()
        os.chdir(ROOT)
        self.addCleanup(os.chdir, self._cwd)
        with open(fp.CSV_FILE, newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def test_csv_ids_unique_and_within_limit(self):
        ids = [row["id"] for row in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(len(ids), fp.MAX_REPOS)

    def test_no_excluded_repo_in_csv(self):
        full_names, names = fp.load_excluded()
        for row in self.rows:
            self.assertNotIn(f"{row['owner']}/{row['name']}".lower(), full_names)
            self.assertNotIn(row["name"].lower(), names)

    def test_logo_files_exist(self):
        with open(fp.LOGO_CSV_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.assertTrue(os.path.exists(row["logo"]), row["logo"])

    def test_readme_matches_csv(self):
        with open(fp.OUTPUT_FILE, encoding="utf-8") as f:
            committed = f.read()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "README.md")
            with mock.patch.object(fp, "OUTPUT_FILE", out):
                fp.convert_csv_to_readme()
            with open(out, encoding="utf-8") as f:
                self.assertEqual(f.read(), committed)


class MainTest(unittest.TestCase):
    """Run main() end to end in a temp directory with the network mocked out."""

    def run_main(self, csv_rows, search_repos, generate):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = os.getcwd()
        os.chdir(tmp.name)
        self.addCleanup(os.chdir, cwd)

        with open(fp.CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "owner", "name", "stars", "url", "business_model"])
            writer.writerows(csv_rows)
        for name, text in [(fp.LOGO_CSV_FILE, "owner,name,logo\n"),
                           (fp.EXCLUDE_FILE, "blocked\n"), ("extra-repos.txt", "")]:
            with open(name, "w") as f:
                f.write(text)

        env = {"GITHUB_TOKEN": "x", "OPENAI_API_KEY": "x"}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(fp, "refresh_existing_repos"), \
                mock.patch.object(fp, "TOPICS", ["ai"]), \
                mock.patch.object(fp, "fetch_repositories", return_value=search_repos), \
                mock.patch.object(fp, "generate_business_model", side_effect=generate) as gen:
            fp.main()
        with open(fp.CSV_FILE, newline="", encoding="utf-8") as f:
            return {int(r["id"]): r for r in csv.DictReader(f)}, gen

    def test_new_fallback_and_generated_repos(self):
        rows = [
            [1, "acme", "done", 30000, "https://github.com/acme/done", "Sell **hosting**."],
            [2, "acme", "stuck", 25000, "https://github.com/acme/stuck", "Plain description"],
        ]
        search = [
            make_repo(1, "done", "Plain"),
            make_repo(2, "stuck", "Plain description"),
            make_repo(3, "fresh", "Brand new"),
            make_repo(4, "blocked", "Excluded by name"),
        ]
        result, gen = self.run_main(rows, search, lambda repo: f"**{repo['name']}** income")

        self.assertEqual(result[1]["business_model"], "Sell **hosting**.")  # untouched
        self.assertEqual(result[2]["business_model"], "**stuck** income")  # fallback retried
        self.assertEqual(result[3]["business_model"], "**fresh** income")  # new repo
        self.assertNotIn(4, result)
        self.assertEqual(gen.call_count, 2)
        with open(fp.OUTPUT_FILE, encoding="utf-8") as f:
            self.assertIn("**stuck** income", f.read())

    def test_openai_failure_keeps_stored_text_and_falls_back(self):
        rows = [[2, "acme", "stuck", 25000, "https://github.com/acme/stuck", "Old text"]]
        search = [make_repo(2, "stuck", "Changed description"), make_repo(3, "fresh", "Brand new")]

        def fail(repo):
            raise RuntimeError("quota")

        result, _ = self.run_main(rows, search, fail)
        self.assertEqual(result[2]["business_model"], "Old text")
        self.assertEqual(result[3]["business_model"], "Brand new")


if __name__ == "__main__":
    unittest.main()
