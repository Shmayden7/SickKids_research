import React, {useState, useEffect} from 'react'
import './App.css'
import { Grid, Paper, Box } from '@mui/material';

function App(){
  const [data, setData] = useState([{}])

  useEffect(() => {
    fetch("http://127.0.0.1:5000/members").then(
      res => res.json()
    ).then(
      data => {
        setData(data)
      }
    )
  }, [])

  return (
    <div className="App">
      <Box sx={{ flexGrow: 1 }}>
        <Grid container spacing={0} alignItems='center' align='center'>
            <Grid item xs={12} sm={6} md={3}>
              <Paper>A</Paper>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Paper>B</Paper>
            </Grid>
        </Grid>
      </Box>
    </div>
  )
}
export default App