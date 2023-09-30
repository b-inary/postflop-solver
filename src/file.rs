// [File format]
// The file consists of a header and a body. The header is as follows:
//  - Magic number (4 bytes): 90 57 f1 09
//  - Version number (1 byte): 1
//  - Compression type (1 byte): 0 (none), 1 (zstd)
//  - Data type (1 byte): 0 (game), 1 (bunching)
//  - Estimated memory usage (`VarIntEncoding`)
//  - Memo string
//
// `VarIntEncoding`: https://github.com/bincode-org/bincode/blob/trunk/docs/spec.md#varintencoding

use crate::bunching::*;
use crate::game::*;
use crate::interface::*;
use bincode::{Decode, Encode};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: u32 = 0x09f15790;
const VERSION: u8 = 1;

#[doc(hidden)]
pub enum DataType {
    Game = 0,
    Bunching = 1,
}

/// A trait for data that can be saved into a file.
pub trait FileData: Decode + Encode {
    #[doc(hidden)]
    fn data_type() -> DataType;
    #[doc(hidden)]
    fn is_ready_to_save(&self) -> bool;
    #[doc(hidden)]
    fn estimated_memory_usage(&self) -> u64;
}

fn encode_into_std_write<E: Encode, W: Write>(
    val: E,
    writer: &mut W,
    err_msg: &str,
) -> Result<usize, String> {
    bincode::encode_into_std_write(val, writer, bincode::config::standard())
        .map_err(|e| format!("{}: {}", err_msg, e))
}

/// Saves data into a standard writer.
///
/// This function serializes the `data` into the `writer`.
/// This is useful if you want to save the data into a custom writer like `Vec<u8>`, but if you want
/// to save the data into a file, use [`save_data_to_file`] instead.
///
/// # Arguments
///
/// - `data`: The data to be saved, which is either a [`PostFlopGame`] or a [`BunchingData`].
/// - `memo`: A memo string to be saved with the data.
/// - `writer`: The writer to write the data into.
/// - `compression_level`: The zstd compression level to use. If `None`, no compression is used.
///   `Some(level)` can only be specified if the `zstd` feature is enabled.
pub fn save_data_into_std_write<T: FileData, W: Write>(
    data: &T,
    memo: &str,
    writer: &mut W,
    compression_level: Option<i32>,
) -> Result<(), String> {
    if !data.is_ready_to_save() {
        return Err("Data is not ready to save".to_string());
    }

    #[cfg(not(feature = "zstd"))]
    if compression_level.is_some() {
        return Err("Compression is not supported".to_string());
    }

    encode_into_std_write(MAGIC, writer, "Failed to write magic number")?;
    encode_into_std_write(VERSION, writer, "Failed to write version number")?;

    let compression_type = compression_level.is_some() as u8;
    encode_into_std_write(compression_type, writer, "Failed to write compression type")?;

    encode_into_std_write(T::data_type() as u8, writer, "Failed to write data type")?;
    encode_into_std_write(
        data.estimated_memory_usage(),
        writer,
        "Failed to write memory usage",
    )?;

    encode_into_std_write(memo, writer, "Failed to write memo")?;

    if compression_level.is_none() {
        encode_into_std_write(data, writer, "Failed to write data")?;
        writer
            .flush()
            .map_err(|e| format!("Failed to flush writer: {}", e))?;
    }

    #[cfg(feature = "zstd")]
    if let Some(compression_level) = compression_level {
        let mut zstd_encoder = zstd::stream::Encoder::new(writer, compression_level)
            .map_err(|e| format!("Failed to create zstd encoder: {}", e))?;

        #[cfg(feature = "rayon")]
        zstd_encoder
            .multithread(rayon::current_num_threads() as u32)
            .map_err(|e| format!("Failed to enable multithreaded zstd encoder: {}", e))?;

        encode_into_std_write(data, &mut zstd_encoder, "Failed to write data")?;
        zstd_encoder
            .finish()
            .map_err(|e| format!("Failed to finish zstd encoder: {}", e))?
            .flush()
            .map_err(|e| format!("Failed to flush writer: {}", e))?;
    }

    Ok(())
}

/// Saves data into a file.
///
/// This function serializes the `data` into a file specified by `path`.
/// If the file already exists, it will be overwritten.
///
/// # Arguments
///
/// - `data`: The data to be saved, which is either a [`PostFlopGame`] or a [`BunchingData`].
/// - `memo`: A memo string to be saved with the data.
/// - `path`: The path to the file to save.
/// - `compression_level`: The zstd compression level to use. If `None`, no compression is used.
///   `Some(level)` can only be specified if the `zstd` feature is enabled.
pub fn save_data_to_file<T: FileData, P: AsRef<Path>>(
    data: &T,
    memo: &str,
    path: P,
    compression_level: Option<i32>,
) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut writer = BufWriter::new(file);
    save_data_into_std_write(data, memo, &mut writer, compression_level)
}

fn decode_from_std_read<D: Decode, R: Read>(reader: &mut R, err_msg: &str) -> Result<D, String> {
    bincode::decode_from_std_read(reader, bincode::config::standard())
        .map_err(|e| format!("{}: {}", err_msg, e))
}

/// Loads data from a standard reader.
///
/// This function deserializes the data from the `reader`.
/// This is useful if you want to load the data from a custom reader like `Vec<u8>`, but if you want
/// to load the data from a file, use [`load_data_from_file`] instead.
///
/// # Arguments
///
/// - `reader`: The reader to read the data from.
/// - `max_memory_usage`: The maximum memory usage allowed for the data (in bytes). If `None`, no
///   limit is set. If the estimated memory usage exceeds this value, `Err` is returned.
///
/// # Returns
///
/// A tuple of the deserialized data (either a [`PostFlopGame`] or a [`BunchingData`]) and the memo
/// string.
pub fn load_data_from_std_read<T: FileData, R: Read>(
    reader: &mut R,
    max_memory_usage: Option<u64>,
) -> Result<(T, String), String> {
    let magic: u32 = decode_from_std_read(reader, "Failed to read magic number")?;
    if magic != MAGIC {
        return Err("Magic number is invalid".to_string());
    }

    let version: u8 = decode_from_std_read(reader, "Failed to read version number")?;
    if version != VERSION {
        return Err("Version number is invalid".to_string());
    }

    let compression_type: u8 = decode_from_std_read(reader, "Failed to read compression type")?;
    if compression_type > 1 {
        return Err("Compression type is invalid".to_string());
    }

    #[cfg(not(feature = "zstd"))]
    if compression_type == 1 {
        return Err("Compression is not supported".to_string());
    }

    let data_type: u8 = decode_from_std_read(reader, "Failed to read data type")?;
    if data_type != T::data_type() as u8 {
        return Err("Data type is invalid".to_string());
    }

    let estimated_memory_usage: u64 = decode_from_std_read(reader, "Failed to read memory usage")?;
    if let Some(max_memory_usage) = max_memory_usage {
        if estimated_memory_usage > max_memory_usage {
            return Err("Estimated memory usage is too large".to_string());
        }
    }

    let memo: String = decode_from_std_read(reader, "Failed to read memo")?;

    #[cfg(not(feature = "zstd"))]
    let data: T = decode_from_std_read(reader, "Failed to read data")?;
    #[cfg(feature = "zstd")]
    let data: T = if compression_type == 0 {
        decode_from_std_read(reader, "Failed to read data")?
    } else {
        let mut zstd_decoder = zstd::stream::Decoder::new(reader)
            .map_err(|e| format!("Failed to create zstd decoder: {}", e))?;
        decode_from_std_read(&mut zstd_decoder, "Failed to read data")?
    };

    Ok((data, memo))
}

/// Loads data from a file.
///
/// This function deserializes the data from a file specified by `path`.
///
/// # Arguments
///
/// - `path`: The path to the file to load.
/// - `max_memory_usage`: The maximum memory usage allowed for the data (in bytes). If `None`, no
///   limit is set. If the estimated memory usage exceeds this value, `Err` is returned.
///
/// # Returns
///
/// A tuple of the deserialized data (either a [`PostFlopGame`] or a [`BunchingData`]) and the memo
/// string.
pub fn load_data_from_file<T: FileData, P: AsRef<Path>>(
    path: P,
    max_memory_usage: Option<u64>,
) -> Result<(T, String), String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut reader = BufReader::new(file);
    load_data_from_std_read(&mut reader, max_memory_usage)
}

impl FileData for PostFlopGame {
    fn data_type() -> DataType {
        DataType::Game
    }

    fn is_ready_to_save(&self) -> bool {
        self.is_solved()
    }

    fn estimated_memory_usage(&self) -> u64 {
        self.target_memory_usage()
    }
}

impl FileData for BunchingData {
    fn data_type() -> DataType {
        DataType::Bunching
    }

    fn is_ready_to_save(&self) -> bool {
        self.is_ready()
    }

    fn estimated_memory_usage(&self) -> u64 {
        self.memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_tree::*;
    use crate::card::*;
    use crate::range::*;
    use crate::utility::*;

    #[test]
    fn save_and_load_file() {
        let card_config = CardConfig {
            range: [Range::ones(); 2],
            flop: flop_from_str("Td9d6h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 970,
            flop_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            turn_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        // save
        save_data_to_file(&game, "", "tmpfile.flop", None).unwrap();

        // load
        let mut game: PostFlopGame = load_data_from_file("tmpfile.flop", None).unwrap().0;

        // save (turn)
        game.set_target_storage_mode(BoardState::Turn).unwrap();
        save_data_to_file(&game, "", "tmpfile.flop", None).unwrap();

        // load (turn)
        let mut game: PostFlopGame = load_data_from_file("tmpfile.flop", None).unwrap().0;

        // save (flop)
        game.set_target_storage_mode(BoardState::Flop).unwrap();
        save_data_to_file(&game, "", "tmpfile.flop", None).unwrap();

        // load (flop)
        let mut game: PostFlopGame = load_data_from_file("tmpfile.flop", None).unwrap().0;

        // remove tmpfile
        std::fs::remove_file("tmpfile.flop").unwrap();

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let root_equity_oop = compute_average(&game.equity(0), weights_oop);
        let root_equity_ip = compute_average(&game.equity(1), weights_ip);
        let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

        assert!((root_equity_oop - 0.5).abs() < 1e-5);
        assert!((root_equity_ip - 0.5).abs() < 1e-5);
        assert!((root_ev_oop - 45.0).abs() < 1e-4);
        assert!((root_ev_ip - 15.0).abs() < 1e-4);
    }

    #[test]
    #[cfg(feature = "zstd")]
    fn save_and_load_file_compressed() {
        let card_config = CardConfig {
            range: [Range::ones(); 2],
            flop: flop_from_str("Td9d6h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 970,
            flop_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            turn_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

        game.allocate_memory(false);
        finalize(&mut game);

        // save
        save_data_to_file(&game, "", "tmpfile-zstd.flop", Some(3)).unwrap();

        // load
        let mut game: PostFlopGame = load_data_from_file("tmpfile-zstd.flop", None).unwrap().0;

        // remove tmpfile
        std::fs::remove_file("tmpfile-zstd.flop").unwrap();

        game.cache_normalized_weights();
        let weights_oop = game.normalized_weights(0);
        let weights_ip = game.normalized_weights(1);
        let root_equity_oop = compute_average(&game.equity(0), weights_oop);
        let root_equity_ip = compute_average(&game.equity(1), weights_ip);
        let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
        let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

        assert!((root_equity_oop - 0.5).abs() < 1e-5);
        assert!((root_equity_ip - 0.5).abs() < 1e-5);
        assert!((root_ev_oop - 45.0).abs() < 1e-4);
        assert!((root_ev_ip - 15.0).abs() < 1e-4);
    }
}
