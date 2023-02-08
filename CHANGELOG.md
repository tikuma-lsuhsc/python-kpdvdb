# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2023-02-07

### Added

- `read_data()` method to read specific recording

### Changed

- the index of the dataframe returned by `query()` is now named `ID`
- **filters argument accepts `ID`
- `type` argument renamed to `task`
- `PAT_ID` shortened of those data without supplied ID

## [0.4.0] - 2023-01-18

### Added

- remove_unknowns xtor argument to drop entries with missing info
- get_diagnoses() include_locations arg to return disorder location

### Fixed

- diagnoses DF to exclude NA entries


### Removed

- KPDVDB.get_locations() method (merged with get_diagnoses())

## [0.3.1] - 2023-01-17

### Fixed
- Fixed diagnoses_filter argument processing

## [0.3.0] - 2023-01-17

### Added
- Added diagnoses_filter argument to query(), get_files(), & iter_data()

## [0.2.1] - 2022-04-06
### Added
- Added `normalize` argument to `KPDVDB.iter_data()` to output float data (now default)
 
## [0.2.0] - 2022-03-31
### Changed
- Combined all the functions to `KPDVDB` class

## [0.1.1] - 2021-12-20
### Fixed
- query() to raise invalid column error.

## [0.1.0] - 2021-12-10
### Added
- Initial release.
